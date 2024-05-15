# from transformers import pipeline
#
# # instantiate pipeline
# classifier = pipeline(
#     "token-classification",
#     model=lora_model,
#     tokenizer=tokenizer,
#     stride=STRIDE,
#     aggregation_strategy='max'
# )
#
# # raises warning: The model 'PeftModelForTokenClassification' is not
# # supported for token-classification. Supported models are
# # ['AlbertForTokenClassification', 'BertForTokenClassification',
# # 'BigBirdForTokenClassification', …
#
#
# def lora_training_config():
#
#     from peft import (
#         get_peft_model,
#         LoraConfig,
#         TaskType
#     )
#
#     # LoRA configuration
#     peft_config = LoraConfig(
#         task_type=TaskType.TOKEN_CLS,
#         inference_mode=False,
#         r=16,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         bias="all",
#         target_modules=["key", "query", "value"],
#         modules_to_save=["classifier"],
#     )
#
#     # load LoRA model and print parameters
#     lora_model = get_peft_model(model, peft_config)
#     lora_model.print_trainable_parameters()
#     # returns: trainable params: 989,186 || all params: 110,370,052 || trainable%: 0.8962449342689446
#
#     # print model details
#     for layer_name, params in lora_model.named_parameters():
#         print(layer_name, params.shape)