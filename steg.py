import torch

def gen_text(gen, num, i2w):
    samples = gen.sample(num)
    texts = []
    for sample in samples:
        text = []
        for word_idx in sample:
            if word_idx == 1:
                break
            text.append(i2w[word_idx.item()])
        texts.append(text)
    with open('image_coco_gen.txt', 'w') as f:
        for text in texts:
            temp = ""
            for word in text:
                temp += word
                temp += ' '
            temp += '\n'
            f.write(temp)

gen = torch.load('gen.model')
gen.eval()
i2w = torch.load('data/image_coco.i2w')
gen_text(gen, 100, i2w)            
