from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial
from sklearn.utils import shuffle
from lxmert.lxrt.tokenization import BertTokenizer
from lxmert.utils import load_obj_tsv

import torch
import torch.utils.data as data

import multiprocessing
import six

generated_caps = {}
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
LABELS = ['NOT_HAL', 'HAL']

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat

class Dataset(data.Dataset):
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.tokenizer.ids_to_tokens

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.tiny = opt.tiny
        self.seq_per_img = opt.seq_per_img
        self.use_hal = getattr(opt, 'use_hal', False)
        gumbel_alpha = getattr(opt, 'gumbel_alpha', -1)
        self.use_gumbel = gumbel_alpha>0
        self.num_hal_per_batch = getattr(opt, 'hal_per_batch', 8)
        self.hal_cap_files = getattr(opt, 'hal_cap_files', 'none')
        if self.use_hal:
            self.opt.hal_prob = self.num_hal_per_batch/(self.opt.batch_size * self.seq_per_img + self.num_hal_per_batch)
            print("hallucination probability: ", self.opt.hal_prob)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.bos_idx = self.tokenizer.vocab["[CLS]"]
        self.eos_idx = self.tokenizer.vocab["[SEP]"]
        self.pad_idx = self.tokenizer.vocab["[PAD]"]
        self.unk_idx = self.tokenizer.vocab["[UNK]"]
        self.vocab_size = len(self.tokenizer.vocab)
        self.opt.special_idx = []
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
        if self.use_hal:
            # add <hal> and <not_hal> token to vocab
            self.tokenizer.add_vocab(["<hal>", "<not_hal>"])
            self.hal_idx = self.tokenizer.vocab["<hal>"]
            self.not_hal_idx = self.tokenizer.vocab["<not_hal>"]
            self.vocab_size = len(self.tokenizer.vocab)
            # add new tokens to opt
            self.opt.hal_idx = self.hal_idx
            self.opt.not_hal_idx = self.not_hal_idx
            self.opt.special_idx += [self.hal_idx, self.not_hal_idx]
        self.opt.vocab_size = self.vocab_size
        print('vocab size is ', self.vocab_size)
        self.opt.bos_idx = self.bos_idx
        self.opt.eos_idx = self.eos_idx
        self.opt.pad_idx = self.pad_idx
        self.opt.unk_idx = self.unk_idx
        self.opt.special_idx += [self.bos_idx, self.eos_idx, self.pad_idx, self.unk_idx]

        print("special token ids: ", self.opt.special_idx)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            # self.seq_length = seq_size[1]*2
            self.seq_length = 20
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1
        self.opt.seq_length = self.seq_length

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)
        # imgid to ix dict
        self.imgid2ix = [(self.info['images'][i]['id'], i) for i in range(len(self.info['images']))]
        self.imgid2ix = dict(self.imgid2ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        # open generated hallucinated caption file
        self.hal_caps = []
        self.hal_imgid = []
        if self.use_hal and self.hal_cap_files != 'none':
            hal_cap_files = self.hal_cap_files.strip().split(',')
            for hal_cap_file in hal_cap_files:
                print("loading ", hal_cap_file, "...")
                with open(generated_caps[hal_cap_file]) as rf:
                    count = 0
                    for i, row in enumerate(rf):
                        # if cap_dict['metrics']['CHAIRs'] > 0 and len(cap_dict['caption']) > 0 and (
                        # self.imgid2ix[cap_dict['image_id']] not in self.split_ix['val']+self.split_ix['test']):
                            # print(self.imgid2ix[cap_dict['image_id']])
                        cap_dict = json.loads(row.strip())
                        cap = cap_dict['caption']
                        seq = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(cap.strip()))
                        seq = [self.hal_idx, self.bos_idx] + seq + [self.eos_idx]
                        seq_length = self.seq_length + 3
                        ids = [self.pad_idx]*seq_length
                        if len(seq) <= seq_length:
                            ids[:len(seq)] = seq
                        else:
                            ids = seq[:seq_length]
                        self.hal_caps.append(ids)
                        self.hal_imgid.append(cap_dict['image_id'])
                        count += 1
                    print("loaded ", count, " captions")
            assert len(self.hal_caps) == len(self.hal_imgid)
            self.num_hal_caps = len(self.hal_caps)
            print("loaded {} hallucinated captions in total.".format(self.num_hal_caps))
            self.hal_img2capix = {}
            self.hal_img2capix_key = []
            for i, imgid in enumerate(self.hal_imgid):
                if imgid not in self.hal_img2capix:
                    self.hal_img2capix[imgid] = [i]
                    self.hal_img2capix_key.append(imgid)
                else:
                    self.hal_img2capix[imgid].append(i)
            self.num_hal_images = len(self.hal_img2capix)
            print("among which there are {} unique hallucinated images in total.".format(self.num_hal_images))
            self.num_hal_cap_per_img = [len(self.hal_img2capix[i]) for i in self.hal_img2capix]
            print("num hal cap per image ranging from ", min(self.num_hal_cap_per_img), ' to ', max(self.num_hal_cap_per_img))
                
        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # Loading detection features to img_data for classification model
        if self.use_gumbel:
            img_data = []
            for split in ['train2014', 'val2014']:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if self.tiny else None
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (split)),
                    topk=load_topk))
            # Convert img list to dict
            self.imgid2img = {}
            for img_datum in img_data:
                imid = int(img_datum['img_id'].strip().split('_')[-1])
                self.imgid2img[imid] = img_datum

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]
        
        seq_ = [[self.ix_to_word[str(i)] for i in cap if i>0] for cap in seq]
        seq_ = [' '.join(cap)+'.' for cap in seq_]
        seq = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(cap.strip()))
            for cap in seq_]

        if self.use_hal:
            seq = [[self.not_hal_idx, self.bos_idx] + list(cap) + [self.eos_idx] for cap in seq]
            num_add_ids = 3
        else:
            seq = [[self.bos_idx] + list(cap) + [self.eos_idx] for cap in seq]
            num_add_ids = 2

        all_ids = []
        seq_length = self.seq_length + num_add_ids
        for cap in seq:
            ids = [self.pad_idx]*seq_length
            if len(cap) <= seq_length:
                ids[:len(cap)] = cap
            else:
                ids = cap[:seq_length]
            all_ids.append(ids)
        return all_ids

    def sample_hal_examples(self):
        ixs = np.random.randint(self.num_hal_caps, size=self.num_hal_per_batch)
        imgids = []
        caps = []
        for i in ixs:
            imgid = self.hal_imgid[i]
            cap = self.hal_caps[i]
            imgids.append(imgid)
            caps.append(cap)        
        return imgids, caps

    def collate_func(self, batch, split):
        fc_batch = []
        att_batch = []
        feat_batch = []
        box_batch = []
        label_batch = []
        target_batch = []

        wrapped = False

        infos = []
        gts = []

        if self.opt.do_train and split == 'train':
            # load generated hallucinated coco img features if needed
            if self.hal_cap_files != 'none':
                img_ids, hal_caps = self.sample_hal_examples()
                for i, cap in zip(img_ids, hal_caps):
                    hal_ix = self.imgid2ix[i]
                    hal_fc_feat, hal_att_feat = self.get_img_features(hal_ix)
                    fc_batch.append(hal_fc_feat)
                    att_batch.append(hal_att_feat)
                    if self.use_gumbel:
                        hal_feats, hal_boxes = self.get_clf_img_feat(hal_ix)
                        feat_batch.append(hal_feats)
                        box_batch.append(hal_boxes)
                        target_batch.append(1)
                    label_batch.append(np.array(cap))

                    if hasattr(self, 'h5_label_file'):
                        # Used for reward evaluation
                        gts.append([self.hal_caps[j] for j in self.hal_img2capix[i]])
                    else:
                        gts.append([])

                    # record associated info as well
                    info_dict = {}
                    info_dict['ix'] = hal_ix
                    info_dict['id'] = self.info['images'][hal_ix]['id']
                    info_dict['file_path'] = self.info['images'][hal_ix].get('file_path', '')
                    infos.append(info_dict)

            for sample in batch:
                # fetch image
                tmp_fc, tmp_att, tmp_feat, tmp_box, tmp_seq, ix, it_pos_now, tmp_wrapped = sample
                if tmp_wrapped:
                    wrapped = True

                for i in range(self.seq_per_img):
                    fc_batch.append(tmp_fc)
                    att_batch.append(tmp_att)
                    label_batch.append(np.array(tmp_seq[i]))
                    if self.use_gumbel:
                        feat_batch.append(tmp_feat)
                        box_batch.append(tmp_box)
                        target_batch.append(0)

                    if hasattr(self, 'h5_label_file'):
                        # Used for reward evaluation
                        gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
                    else:
                        gts.append([])

                    # record associated info as well
                    info_dict = {}
                    info_dict['ix'] = ix
                    info_dict['id'] = self.info['images'][ix]['id']
                    info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
                    infos.append(info_dict)
        else:
            for sample in batch:
                # fetch image
                tmp_fc, tmp_att, tmp_feat, tmp_box, tmp_seq, ix, it_pos_now, tmp_wrapped = sample
                if tmp_wrapped:
                    wrapped = True

                fc_batch.append(tmp_fc)
                att_batch.append(tmp_att)
                label_batch.append(np.array(tmp_seq))
                if self.use_gumbel:
                    feat_batch.append(tmp_feat)
                    box_batch.append(tmp_box)
                    target_batch.append(0)

                # Used for reward evaluation
                if hasattr(self, 'h5_label_file'):
                    # if there is ground truth
                    gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
                else:
                    gts.append([])
        
                # record associated info as well
                info_dict = {}
                info_dict['ix'] = ix
                info_dict['id'] = self.info['images'][ix]['id']
                info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
                infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        if self.use_gumbel:
            fc_batch, att_batch, feat_batch, box_batch, label_batch, gts, infos, target_batch = shuffle(
                fc_batch, att_batch, feat_batch, box_batch, label_batch, gts, infos, target_batch, random_state=1234)
                # zip(*sorted(zip(fc_batch, att_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
        else:
            fc_batch, att_batch, label_batch, gts, infos = shuffle(
                fc_batch, att_batch, label_batch, gts, infos, random_state=1234)

        data = {}
        data['fc_feats'] = np.stack(fc_batch)
        if self.use_gumbel:
            data['feat'] = np.stack(feat_batch)
            data['pos'] = np.stack(box_batch)
            data['target'] = np.array(target_batch)
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        if self.opt.do_train and split == 'train':
            data['labels'] = np.stack(label_batch)
            # generate mask
            if self.use_hal:
                nonzeros = np.array(list(map(lambda x: (x != self.pad_idx).sum() + 1, data['labels'])))
                mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 3], dtype = 'float32')
            else:
                nonzeros = np.array(list(map(lambda x: (x != self.pad_idx).sum() + 1, data['labels'])))
                mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
            
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            data['masks'] = mask_batch
        else:
            data['labels'] = np.vstack(label_batch)
            # generate mask
            if self.use_hal:
                nonzeros = np.array(list(map(lambda x: (x != self.pad_idx).sum() + 1, data['labels'])))
                mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 3], dtype = 'float32')
            else:
                nonzeros = np.array(list(map(lambda x: (x != self.pad_idx).sum() + 1, data['labels'])))
                mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')

            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            data['masks'] = mask_batch
            data['labels'] = data['labels'].reshape(len(batch), self.seq_per_img, -1)
            data['masks'] = data['masks'].reshape(len(batch), self.seq_per_img, -1)

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def get_img_features(self, ix):
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                # devided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((0,0), dtype='float32')
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        
        return fc_feat, att_feat

    def get_clf_img_feat(self, ix):
        img_id = self.info['images'][ix]['id']
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        return feats, boxes

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index #self.split_ix[index]

        # load origianl coco img features
        fc_feat, att_feat = self.get_img_features(ix)
        if self.use_gumbel:
            feats, boxes = self.get_clf_img_feat(ix)
        else:
            feats, boxes = None, None
        # load original coco captions
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
        return (fc_feat, att_feat, feats, boxes, seq, ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['images'])

class DataLoader:
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.dataset = Dataset(opt)
        self.opt = self.dataset.opt

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=4, # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func, split=split),
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }

    