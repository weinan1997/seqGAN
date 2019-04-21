import torch

def gen_text(gen, num, i2w):
    samples = gen.sample(num)
    texts = []
    for sample in samples:
        text = []
        for word_idx in sample:
            if word_idx == 1:
                break
            text.append(i2w[word_idx])
        texts.append(text)
    with open('image_coco_gen.txt', 'w') as f:
        for text in texts:
            temp = ""
            for word in text:
                temp += word
                temp += ' '
            f.write(temp)