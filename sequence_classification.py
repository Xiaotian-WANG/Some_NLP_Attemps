from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
inputs = tokenizer("伊朗民航局（CAO）公布了此次事故的最终调查报告，总结坠机事故发生的可能原因如下： 1. 在飞机离地前约2秒，2号发动机停车的同时电子发动机控制系统（SAY-2000型）发生失效。2. 飞机飞行手册（AFM）中混乱的性能图表导致飞行员所依靠的（飞机）性能计算过高地估计了飞机的最大起飞重量（MTOW）。事故的影响因素包括： 1. 飞机飞行手册（AFM）中的程序不清楚，包括计算最大允许起飞重量、抬前轮速度（VR）和起飞安全速度（V2）的程序，并且对爬升段的定义和应用比较模糊。2. 机组成员的表现，包括： 机长（PIC）以大约219公里/小时的速度抬前轮（而飞机飞行手册中表4.2.3建议的速度为224公里/小时） 机组成员未能对失效的发动机执行手动螺旋桨顺桨程序。尽管飞机超重约190千克，机长仍决定起飞。发生事故的飞机所带燃油比所需燃油多500千克左右。在飞机的审定试飞期间，没有考虑起飞螺旋桨叶片未顺桨导致的负推力，因为（审定人员）认为这一情况是不可能发生的。然而，在事故发生时，负推力确实出现并影响了飞机性能。", return_tensors="pt")
labels = torch.tensor([0]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits