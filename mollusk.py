#predicting mollusk sex

# https://wiznod.com/job-apply.php?id=5331
import numpy as np
import pdb,os
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import cross_validation
from sklearn import preprocessing
poly=preprocessing.PolynomialFeatures(interaction_only=True)
from sklearn.metrics import confusion_matrix

fname='mollusks.csv'
f=open(fname,'rb')
r=csv.reader(f)
fields=r.next()
all_data=[]
for as_str in r:
	sample=[int(as_str[0])]
	sample.extend([float(el) for el in as_str[1:-2]])
	sample=sample+[int(as_str[-2])]
	sample=sample+[as_str[-1]]
	all_data.append(sample)
	pass

sex=[row[-1] for row in all_data]
sex_dict={'M':0,'F':1,'J':2}
train_valid_y=[sex_dict[row] for row in sex]
train_valid_x=[row[1:-1] for row in all_data]
scale_obj=preprocessing.StandardScaler()
scale_obj.fit(train_valid_x)
train_valid_x=scale_obj.transform(train_valid_x)
train_valid_x=poly.fit_transform(train_valid_x)

from sklearn.utils import shuffle
train_valid_x,train_valid_y=shuffle(train_valid_x,train_valid_y,random_state=0)

train_x,valid_x,train_y,valid_y=cross_validation.train_test_split(train_valid_x,train_valid_y,test_size=0.4)
train_y=np.array(train_y)
train_x=np.array(train_x)
f.close()
# # # # # # # # # ## #  # # # # # # # # # # # # 
# # V I S U A L I Z A T I O N


# pca=PCA()
# pca.fit(train_x)
# projected=pca.transform(train_x)
# projected_2D=projected[:,:2]
# for row in range(len(train_y)):
# 	if train_y[row]==0:
# 		clr='.r'
# 	elif train_y[row]==1:
# 		clr='.g'
# 	elif train_y[row]==2:
# 		clr='.b'

# 	plt.plot(projected_2D[row,0],projected_2D[row,1],clr)
# 	plt.hold(True)
# plt.show()




# plt.plot(projected_2D[:,0],projected_2D[:,1],'.g')
# plt.hold(True)
# plt.show()

# # # # # # # # # ## #  # # # # # # # # # # # # 
# S V M


classifier=OneVsRestClassifier(SVC(kernel='linear',class_weight={0:0.82})).fit(train_x,train_y)


# # # # # # # # # ## #  # # # # # # # # # # # # 
# # D E C I S I O N  T R E E S

# from sklearn import tree
# classifier=tree.DecisionTreeClassifier(max_depth=3,
# 										min_samples_leaf=100).fit(train_x,train_y)

# # # # # # # # # # # #  # # # # # # # # # # # # 
# # R A N D O M  F O R E S T

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=2000,n_jobs=-1).fit(train_x,train_y)


pred_train_y=classifier.predict(train_x)
train_errors=np.where((pred_train_y-train_y)!=0)[0]
n_train_errors=len(train_errors)

pred_valid_y=classifier.predict(valid_x)
valid_errors=np.where((pred_valid_y-valid_y)!=0)[0]
n_valid_errors=len(valid_errors)
print 'classifier train error %d/%d: validation error %d/%d:' %(n_train_errors,len(train_y),n_valid_errors,len(valid_y))

cm_train = confusion_matrix(train_y, pred_train_y)
cm_valid = confusion_matrix(valid_y, pred_valid_y)

plt.matshow(cm_train)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.close()

plt.matshow(cm_valid)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
plt.close()

# # # # # # # # # # # #  # # # # # # # # # # # # 

test_fname='mollusks-todo.csv'
test_f=open(test_fname,'rb')
test_r=csv.reader(test_f)
dummy=test_r.next()
test_data=[]
for as_str in test_r:
	sample=[int(as_str[0])]
	sample.extend([float(el) for el in as_str[1:-2]])
	sample=sample+[int(as_str[-2])]
	sample=sample+[as_str[-1]]
	test_data.append(sample)
	pass
test_ids=[row[0] for row in test_data]
test_x=[row[1:-1] for row in test_data]
test_x=scale_obj.transform(test_x)
test_x=poly.fit_transform(test_x)
test_f.close()
test_y=classifier.predict(test_x)
test_y=np.array(test_y)
test_x=np.array(test_x)

# # # # # # # # # # # #  # # # # # # # # # # # # 

op_fname='mollusk_pred.csv'
op_f=open(op_fname,'wb')
op_w=csv.writer(op_f,delimiter=',')
reverse_dict={val:tag for tag,val in sex_dict.items()}
for row in range(len(test_y)):
	op_w.writerow([int(test_ids[row]),reverse_dict[test_y[row]]])
##
op_f.close()
