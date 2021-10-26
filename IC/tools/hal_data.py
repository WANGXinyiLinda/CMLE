# open generated hallucinated caption file
hal_caps = []
hal_imgid = []
if  hal_cap_files != 'none':
    hal_cap_files =  hal_cap_files.strip().split(',')
    for hal_cap_file in hal_cap_files:
        print("loading ", hal_cap_file, "...")
        caps = json.load(open(generated_caps[hal_cap_file]))['sentences']
        count = 0
        for i, cap_dict in enumerate(caps):
            if cap_dict['metrics']['CHAIRs'] > 0 and len(cap_dict['caption']) > 0:
                cap = cap_dict['caption']
                seq = [ word_to_ix[w] for w in cap.strip().split() if w in word_to_ix]
                ids = [0]* seq_length
                if len(seq) <=  seq_length:
                    ids[:len(seq)] = seq
                else:
                    ids = seq[: seq_length]
                 hal_caps.append(ids)
                 hal_imgid.append(cap_dict['image_id'])
                count += 1
        print("loaded ", count, " captions")
assert len(hal_caps) == len(hal_imgid)
num_hal_caps = len(hal_caps)
print("loaded {} hallucinated captions in total.".format(num_hal_caps))
hal_img2capix = {}
for i, imgid in enumerate( hal_imgid):
    if imgid not in  hal_img2capix:
        hal_img2capix[imgid] = [i]
    else:
        hal_img2capix[imgid].append(i)
num_hal_images = len(hal_img2capix)
print("among which there are {} unique hallucinated images in total.".format( num_hal_images))