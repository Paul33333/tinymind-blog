---
title: reasoning能力微调实战篇：微调qwen基座模型具备思考、推理能力（类似o1）
date: 2024-10-14T12:48:55.285Z
---

## 1、引言
关于openai新发布的o1的特性，本篇帖子就不详细介绍了，其底层想法可以阅读这篇论文：[Let’s Verify Step by Step](https://arxiv.org/pdf/2305.20050 "let's verify step by step")。

本次我们的 **reasoning能力微调实战**的方案如下：
- **数据集**：**reasoning-base-20k**（数据集地址：[KingNish/reasoning-base-20k](https://huggingface.co/datasets/KingNish/reasoning-base-20k "KingNish/reasoning-base-20k")）
- **基座模型**：**qwen2.5-1.5B** （模型地址：[Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B "Qwen/Qwen2.5-1.5B")）
- **微调方法**：SFT**全参**微调；需要特别指出的是微调环节相比**标准的sft微调框架**（标准的sft微调框架可以参看本人之前的一篇分享：[监督式微调(SFT) & 偏好对齐(DPO)：From Zero To Hero](https://zhuanlan.zhihu.com/p/715250294 "监督式微调(SFT) & 偏好对齐(DPO)：From Zero To Hero")）虽然整体思路基本一致，但还是需要注意以下差异点：
 - 1、需要增加`reasoning`这个special token
 - 2、新增`reasoning`special token后，还需要同步调整模型的`embedding`层
 - 3、还需同步调整聊天模版（聊天模板的重要性请参阅这篇文章：[Chat Templates](https://hf-mirror.com/blog/chat-templates "Chat Templates")）
 - 4、计算`Loss`（损失）的时候，需要综合考虑reasoning + assistant的部分（或者分开单独训练一个reasoning模型+assistant模型也可，本文未单独训练，将reasoning + assistant的损失统一考虑）
- **算力资源**：google colab的一张A100显卡（方便大家复现）

------------

## 2、reasoning微调实战
比较关键的部分实现如下：
- 1、需要增加`reasoning`这个special token
```python
new_special_token = "<|reasoning|>"
tokenizer.add_special_tokens({"additional_special_tokens": [new_special_token]})
```

- 2、新增`reasoning`special token后，还需要同步调整模型的`embedding`层
```python
model.resize_token_embeddings(len(tokenizer))
```

- 3、同步调整聊天模版——适配`<|reasoning|>`部分
```python
reasoning_data['train'] = reasoning_data['train'].map(lambda x: {**x,
    'user_template': "".join(["<|im_start|>user\n", x['user'], "<|im_end|>\n"]),
    'reasoning_template': "".join(["<|im_start|><|reasoning|>\n", x['reasoning'], "<|im_end|>\n"]),
    'assistant_template': "".join(["<|im_start|>assistant\n", x['assistant'], "<|im_end|>\n"]),
    'template_new': "".join(["<|im_start|>system\nYou are a helpful assistant<|im_end|>\n","<|im_start|>user\n", x['user'], "<|im_end|>\n","<|im_start|><|reasoning|>\n", x['reasoning'], "<|im_end|>\n","<|im_start|>assistant\n", x['assistant'], "<|im_end|>\n"])
})
```

- 4、计算`Loss`（损失）的时候，需要考虑reasoning的部分，这部分还是通过**掩码机制**实现的，对这部分有困惑的可以阅读[监督式微调(SFT) & 偏好对齐(DPO)：From Zero To Hero](https://zhuanlan.zhihu.com/p/715250294 "监督式微调(SFT) & 偏好对齐(DPO)：From Zero To Hero")第2.2节
```python
# 设置问题部分的掩码函数，用于执行仅针对回答部分（此处默认包含reasoning部分）才计算损失
def return_answer_mask(input_ids):
  assistant_answer_mask = torch.zeros_like(input_ids) #0初始化
  for i in range(input_ids.shape[0]):
        ## user部分的结尾\n: \n是<|im_end|>的下一个元素，所以有+1 【这个地方需要根据不同模型的不同聊天模版自定义更改】，关于聊天模版可阅读这篇文章：https://huggingface.co/blog/chat-templates
        i_user_end_list = [i+1 for i in torch.where(input_ids[i]==tokenizer.encode('<|im_end|>')[0])[0].tolist()[1::3]]   #第1个im_end开始
        ## assistant部分的结尾\n：\n是<|im_end|>的下一个元素，所以有+1 【这个地方需要根据不同模型的不同聊天模版自定义更改】
        i_assistant_end_list = [i+1 for i in torch.where(input_ids[i]==tokenizer.encode('<|im_end|>')[0])[0].tolist()[3::3]] #第3个im_end开始

        if len(i_user_end_list)==len(i_assistant_end_list):
            for user_end, assistant_end in zip(i_user_end_list, i_assistant_end_list):
                assistant_answer_mask[i][user_end+3:assistant_end-1]=1 #+3的操作，【这个地方需要根据不同模型的不同聊天模版自定义更改】
        elif len(i_user_end_list)==len(i_assistant_end_list)+1==1:  ##单轮问答,且回答部分未结尾就被截断了
            assistant_answer_mask[i][i_user_end_list[0]+3:]=1  ##会把右补的padding token也标记为1，所以后面还需要再结合padding mask以过滤padding
        elif len(i_user_end_list)==len(i_assistant_end_list)+1:   ##兼顾多轮问答
            assistant_answer_mask[i][i_user_end_list[-1]+3:]=1
            for user_end, assistant_end in zip(i_user_end_list[:-1], i_assistant_end_list):
                assistant_answer_mask[i][user_end+3:assistant_end-1]=1
        else:
            continue  ##跳出当前循环，继续下一次循环
  return assistant_answer_mask
  ```

关键的区别点讲完了，其他废话就不多说了，详细微调代码请参看：
[reasoning能力微调实战篇-微调qwen基座模型具备reasoning思考、推理能力（类似o1）](https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/reasoning%E8%83%BD%E5%8A%9B%E5%BE%AE%E8%B0%83%E5%AE%9E%E6%88%98%E7%AF%87_%E5%BE%AE%E8%B0%83qwen%E5%9F%BA%E5%BA%A7%E6%A8%A1%E5%9E%8B%E5%85%B7%E5%A4%87reasoning%E6%80%9D%E8%80%83%E3%80%81%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%EF%BC%88%E7%B1%BB%E4%BC%BCo1%EF%BC%89.ipynb "reasoning能力微调实战篇-微调qwen基座模型具备reasoning思考、推理能力（类似o1）")

- 这里需要特别指出：因为训练数据集主要是推理数据集，其数据风格比较统一（数据丰富度不够），训练过程能比较快的收敛，但是应该一定的过拟合现象，所以训练过程中损失才下降的这么低

![损失记录.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-14/1728895712564.png?raw=true)

------

## 3、reasoning效果测试
测试模型的reasoning效果时，对比原始的直接推理`assistant`部分的回答，新的推理方案应为两步法（虽然训练环节时，我们采用了一步法统一考虑reasoning和assiatant的损失），具体如下：
- 1、先基于问题推理reasoning回答
- 2、reasoning完成后，将question和reasoning的回答一并再作为**input**送进去模型生成思考后的回答（assistant answer）

具体实现如下：
```python
from IPython.display import Markdown, display
history = []
history.append({"role": "system", "content": "You are a helpful assistant"})
while True:
    question = input('User：' + '\n')
    print('\n')
    history.append({"role": "user", "content": question})
    input_text = new_apply_chat_template(
            history,
            add_reasoning_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    if model_inputs.input_ids.size()[1]>32000:
        break
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=3000
    )
    if len(generated_ids)>32000:
        break
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    reasoning_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history.append({"role": "<|reasoning|>", "content": reasoning_response})
    print('reasoning:\n')
    #print(response)
    display(Markdown(reasoning_response))
    print("------------")
    print('\n')
    input_text = new_apply_chat_template(
            history,
            add_assistant_generation_prompt=True
        )
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    if model_inputs.input_ids.size()[1]>32000:
        break
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=3000
    )
    if len(generated_ids)>32000:
        break
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    assistant_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    history.append({"role": "assistant", "content": assistant_response})
    print('assistant:\n')
    display(Markdown(assistant_response))
    print("------------")
print("超过模型字数上线，已退出")
```

测试效果如下：

![微调模型推理能力1.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-14/1728896138728.png?raw=true)

![微调模型推理能力2.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-14/1728896152137.png?raw=true)

可以看到模型在回答**找出0-10内的所有质数**这个问题时的reasoning详细推理步骤（推理思维链真长...）


