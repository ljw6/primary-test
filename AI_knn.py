import numpy as np
import collections as cl
import random
def colo_to_num(str):
	dic={'红':0.50,'黄':0.51,'蓝':0.52,'绿':0.53,'紫':0.54,'粉':0.55}
	return dic[str]
def knn2(k,predictpoint,ballcolor,feature,lable):
	#此处使用数据归一化
	distance=list(map(lambda item: ((item[0]/475-predictpoint/475)**2+((item[1]-0.5)/0.05-(ballcolor-0.5)/0.05)**2)**0.5, feature))
	sortindex=(np.argsort(distance))
	sortlabel=(lable[sortindex])
	sortlabel=sortlabel.astype(np.int16)
	return (cl.Counter(sortlabel[0:k]).most_common(1)[0][0])
if __name__ == '__main__':
	data=np.loadtxt('train1.csv', delimiter=',',converters={1: colo_to_num},encoding="utf-8-sig")
	feature=(data[:,0:2])
	lable=(data[:,2])
	k=np.sqrt(len(feature))
	count=0
	testdata=np.loadtxt('train1-test.csv', delimiter=',',converters={1: colo_to_num},encoding="utf-8-sig")
	for item in testdata:
		reallabel=item[-1]
		t=knn2(int(k),item[0],item[1],feature,lable)
		if reallabel==t:
			count+=1
	print("准确率为 {}%".format(count*100/len(testdata)))
