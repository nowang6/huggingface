from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
encoded = tokenizer.encode("编码一句话")
print(encoded)
# [101, 5356, 4772, 671, 1368, 6413, 102]
decoded = tokenizer.decode(out)
print(decoded)
#[CLS] 编 码 一 句 话 [SEP]


sents1 = "你站在桥上看风景" 
sents2 = "看风景的人在楼上看你" 
encoded = tokenizer.encode(sents1, sents2)
print(encoded)
# [101, 872, 4991, 1762, 3441, 677, 4692, 7599, 3250, 102, 4692, 7599, 3250, 4638, 782, 1762, 3517, 677, 4692, 872, 102]
decoded = tokenizer.decode(encoded)
print(decoded)


encoded = tokenizer(sents1,
                           sents2,
                           #当句子大于max_length时候，截断
                           truncation = True,
                           #补齐到max_length
                           padding = "max_length",
                           add_special_tokens=True,
                           max_length = 25,
                           return_tensors = None,
                           return_token_type_ids = True,
                           return_attention_mask = True,
                           return_special_tokens_mask = True
                           )
for k, v in encoded.items():
  print(k, ":", v)

# input_ids : [101, 872, 4991, 1762, 3441, 677, 4692, 7599, 3250, 102, 4692, 7599, 3250, 4638, 782, 1762, 3517, 677, 4692, 872, 102, 0, 0, 0, 0]
# 第二个句子为1， 其它句子为0
# token_type_ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# PAD位置为0
# attention_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# special_tokens_mask : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

