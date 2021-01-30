
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.contrib import rnn
import time
import csv
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import os
from numpy import array
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.stats as ss
from scipy.stats.stats import pearsonr  
from math import log

from BKT import BKT

model_name = 'BKT-LSTM'


# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.3, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate",1e-2, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.6, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "The number of hidden layers (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("problem_len", 20, "length for each time interval")
tf.flags.DEFINE_integer("num_cluster", 7, "length for each time interval")
tf.flags.DEFINE_integer("epochs", 40, "Number of epochs to train for.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("model_name", model_name, "model used")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()

def add_gradient_noise(t, stddev=1e-3, name=None):

    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class StudentModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size
        self.num_skills = num_skills = config.num_skills 
        self.num_steps = num_steps = config.num_steps
        
        
        label_size = (num_skills*2)
        id_size = num_skills 
        df_size = 11
        cluster_size=(FLAGS.num_cluster+1)        
        reuse_flag=False
        
        output_size = (num_skills+1)
        
        
        self.cluster = tf.placeholder(tf.int32, [batch_size, num_steps], name='profile')
        self.diff = tf.placeholder(tf.int32, [batch_size, num_steps], name='difficulty')
        self.assess = tf.placeholder(tf.float32, [batch_size, num_steps], name='Assessment')
        
        self._target_id = target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])
        
        
        #one-hot encoding
        
        cluster = tf.reshape(self.cluster, [-1])
        slice_cluster_data = one_hot_output(cluster, cluster_size, batch_size, num_steps)
        
        diff = tf.reshape(self.diff, [-1])
        slice_diff_data = one_hot_output(diff, df_size, batch_size, num_steps)
        
        assess = tf.reshape(self.assess, [batch_size, num_steps, 1])
        slice_ass_data = tf.split(assess, num_steps, 1)
        
        input_l = []
        for i in range(num_steps): 
            if i != 0:
               reuse_flag = True
                        
            
            cu = tf.squeeze(slice_cluster_data[i], 1)
            df = tf.squeeze(slice_diff_data[i], 1)
            ass =  tf.squeeze(slice_ass_data[i], 1)
            ## Concat [Assessment, Profile, Difficulty]
            m1 = tf.concat([ass, cu, df], 1) 
            input_l.append(m1)            
            
            
            
        input_= tf.stack(input_l)        
        input_size=int(input_[0].get_shape()[1])
        x_input = tf.reshape(input_, [-1, input_size])        
        x_input = tf.split(x_input, num_steps, 0)
        
        final_hidden_size = input_size        
        hidden_layers = []
        for i in range(FLAGS.hidden_layer_num):
            final_hidden_size = final_hidden_size
            hidden1 = tf.nn.rnn_cell.LSTMCell(final_hidden_size, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                hidden1 = tf.nn.rnn_cell.DropoutWrapper(hidden1, output_keep_prob=FLAGS.keep_prob)
            hidden_layers.append(hidden1)

        cell = tf.nn.rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)
        
        
        
        outputs, state = rnn.static_rnn(cell, x_input, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs,1), [-1, int(final_hidden_size)])
        sigmoid_w = tf.get_variable("sigmoid_w", [final_hidden_size, output_size])
        sigmoid_b = tf.get_variable("sigmoid_b", [output_size])
        logits = tf.matmul(output, sigmoid_w) + sigmoid_b
        logits = tf.reshape(logits, [-1])        
        selected_logits = tf.gather(logits, self.target_id)
        self._all_logits = tf.sigmoid(logits)

        #make prediction
        self._pred = tf.sigmoid(selected_logits)

        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))
        self._cost = cost = loss

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_data(self):
        return self._input_data

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def all_logits(self):
        return self._all_logits

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

class HyperParamsConfig(object):
    """Small config."""
    init_scale = 0.05    
    num_skills = 0
    num_steps = FLAGS.problem_len
    batch_size = FLAGS.batch_size
    max_grad_norm = FLAGS.max_grad_norm
    max_max_epoch = FLAGS.epochs
    keep_prob = FLAGS.keep_prob
        


def k_means_clust(session, train_students, test_students, max_stu, max_seg, num_clust, num_skills, num_iter):
    identifiers=3
    max_stu=int(max_stu)
    max_seg=int(max_seg)
    cluster= np.zeros((max_stu,max_seg))
    data=[]
    for ind,i in enumerate(train_students):
        data.append(i[:-identifiers])
    data = array(data)
    points = tf.constant(data)  

    centroids = tf.Variable(tf.random_shuffle(points)[:num_clust, :])
    # calculate distances from the centroids to each point
    points_e = tf.expand_dims(points, axis=0) # (1, N, 2)
    centroids_e = tf.expand_dims(centroids, axis=1) # (k, 1, 2)  
    distances = tf.reduce_sum((points_e - centroids_e) ** 2, axis=-1) # (k, N)
    # find the index to the nearest centroids from each point
    indices = tf.argmin(distances, axis=0) # (N,)
    # gather k clusters: list of tensors of shape (N_i, 1, 2) for each i
    clusters = [tf.gather(points, tf.where(tf.equal(indices, i))) for i in range(num_clust)]
    # get new centroids (k, 2)
    new_centroids = tf.concat([tf.reduce_mean(clusters[i], reduction_indices=[0]) for i in range(num_clust)], axis=0)
    # update centroids
    assign = tf.assign(centroids, new_centroids)
    session.run(tf.global_variables_initializer())
    for j in range(num_iter):
        clusters_val, centroids_val, _ = session.run([clusters, centroids, assign])
        
    
    for ind,i in enumerate(train_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None            
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j])< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j])
               if cur_dist<min_dist:                  
                  min_dist=cur_dist
                  closest_clust=j                  
        
        cluster[int(i[-2]),int(i[-1])]=closest_clust
        
   
    for ind,i in enumerate(test_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None 
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j])< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j])
               if cur_dist<min_dist:
                  min_dist=cur_dist
                  closest_clust=j
        cluster[int(i[-2]),int(i[-1])]=closest_clust
        
        
    del train_students, test_students
    
    return cluster
    

    


def one_hot_output(input_data, output_size, batch_size, num_steps):
    with tf.device("/cpu:0"):
         labels = tf.expand_dims(input_data, 1)
         indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
         concated = tf.concat([indices, labels],1)
         input_data = tf.sparse_to_dense(concated, tf.stack([batch_size*num_steps, output_size]), 1.0, 0.0)
         input_data.set_shape([batch_size*num_steps, output_size])
         input_data = tf.reshape(input_data, [batch_size, num_steps, output_size ])
         output_data = tf.split(input_data, num_steps, 1)
    return output_data


  
    
def difficulty_data(students,max_items):
          
    limit= 3
    xtotal = np.zeros(max_items+1)
    x1 = np.zeros(max_items+1)
    items=[]
    Allitems=[]
    item_diff ={} 
    index=0      
    while(index < len(students)):
         student = students[index]         
         item_ids = student[3]
         correctness = student[2]         
         for j in range(len(item_ids)):         
             
             key =item_ids[j]             
             xtotal[key] +=1
             if(int(correctness[j]) == 0):
                x1[key] +=1
             if xtotal[key]>limit and key > 0 and key not in items  :
                items.append(key)
             
             if xtotal[key]>0 and key not in Allitems :
                Allitems.append(key)
                
         index+=1
    for i in (items):
        diff =(np.around(float(x1[i])/float(xtotal[i]), decimals=1)*10).astype(int)   
        item_diff[i]=diff
    

    return item_diff     



def run_epoch(session, m, students, item_diff, max_stu, cluster, eval_op, num_skills, datatype, epoch, verbose=False):
    """Runs the model on the given data."""   
    index = 0
    pred_labels = []
    actual_labels = []
    all_all_logits = []
    check_point= verbose 
        
    while(index+m.batch_size < len(students)):
        cl = np.zeros((m.batch_size, m.num_steps))
        clus = np.zeros((m.batch_size, m.num_steps))
        diff = np.zeros((m.batch_size, m.num_steps))
        assess=np.zeros((m.batch_size, m.num_steps))
       
        
        target_ids = []
        target_correctness = []        
        for i in range(m.batch_size):
            student = students[index+i]
            student_id = student[0][0]
            seg_id = int(student[0][1])             
            
            if (seg_id>0):
                cluster_id= cluster[student_id,(seg_id-1)]+1
            else:
                cluster_id= 0           
            
            problem_ids = student[1]
            correctness = student[2] 
            items = student[3]       
            bkt= student[4]   
                        
            for j in range(len(problem_ids)-1):
                indx= j 
                
                
                target_id = int(problem_ids[indx])
                item = int(items[indx]) 
                kcass = float(bkt[indx])  
                
                current_correct = int(correctness[indx])

                # to ignore if target_id is null or -1 all skill index are started from 0
                
                if target_id > -1:
                
                   df = 0
                   if item in item_diff.keys():                      
                      df = int(item_diff[item])
                   else:
                        df=5
                                     
                   clus[i,j] = cluster_id  ## Ability Profile
                   diff[i,j] = df          ## Problem Difficulty
                   assess[i,j] = np.round(kcass,3)   ## Mastery Assessment  
                   
                   output_size = (m.num_skills+1)                   
                   burffer_space=i*m.num_steps*(output_size)+j*(output_size)
                   t_ind=burffer_space+ int(target_id)
                   target_ids.append(t_ind)
                   
                    
                   target_correctness.append(int(correctness[indx]))                
                   actual_labels.append(int(correctness[indx]))
                 
        index += m.batch_size
        
        
        
        pred, _, all_logits = session.run([m.pred, eval_op, m.all_logits], feed_dict={
            m.cluster: clus, m.diff: diff, m.assess:assess, m.target_id: target_ids, m.target_correctness: target_correctness})
        
        predicted=[]
        for i, p in enumerate(pred):
            pred_labels.append(np.nan_to_num(p))
            predicted.append(float(np.nan_to_num(p)))

        all_all_logits.append(all_logits)
    
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_labels)
    
    pred_labels=np.array(pred_labels)
    pred_labels[pred_labels > 0.5] = 1
    pred_labels[pred_labels <= 0.5] = 0
    acc=metrics.accuracy_score(actual_labels, pred_labels)
    acc=metrics.accuracy_score(actual_labels, pred_labels)
    del students
    return rmse, auc, r2, acc, np.concatenate(all_all_logits)


def read_data_from_csv_file(trainfile, testfile):
    rows = []
    max_skills = 0
    max_steps = 0 
    max_items =0
    studentids = []
    train_ids=[]
    test_ids=[]
    
    problem_len = 20  
    with open(trainfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
            
            
    skill_rows=[]
    correct_rows=[]
    stu_rows=[]
    opp_rows=[]
    index = 0
    while(index < len(rows)):
         if int(rows[index][0])>problem_len: 
            problems = int(rows[index][0]) 
            student_id= int(rows[index][1])
            train_ids.append(student_id)  
            
            tmp_max_skills = max(map(int, rows[index+1]))
            if(tmp_max_skills > max_skills):
               max_skills = tmp_max_skills
                        
                        
            tmp_max_items = max(map(int, rows[index+2]))
            if(tmp_max_items > max_items):
               max_items = tmp_max_items
               
            skill_rows=np.append(skill_rows,rows[index+1])
            correct_rows=np.append(correct_rows,rows[index+3])
            stu_rows=np.append(stu_rows,([student_id]* len(rows[index+1])))
            opp_rows=np.append(opp_rows, list(range(len(rows[index+1]))))
         index += 4  
         
         
         
    with open(testfile, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
            
            
    
    while(index < len(rows)):
         if int(rows[index][0])>problem_len: 
            problems = int(rows[index][0]) 
            student_id= int(rows[index][1])
            test_ids.append(student_id)  
            
            tmp_max_skills = max(map(int, rows[index+1]))
            if(tmp_max_skills > max_skills):
               max_skills = tmp_max_skills
                        
                        
            tmp_max_items = max(map(int, rows[index+2]))
            if(tmp_max_items > max_items):
               max_items = tmp_max_items
               
            skill_rows=np.append(skill_rows,rows[index+1])
            correct_rows=np.append(correct_rows,rows[index+3])
            stu_rows=np.append(stu_rows,([student_id]* len(rows[index+1])))
            opp_rows=np.append(opp_rows, list(range(len(rows[index+1]))))
         index += 4  
         
         
         
         
         
         
    max_skills =max_skills+1
    max_items = max_items+1     
    
    data= pd.DataFrame({'stus': stu_rows, 'skills': skill_rows, 'corrects': correct_rows, 'opp': opp_rows}).astype(int)
    bkt_ass= BKTAssessment(data, train_ids, max_skills)
    
    del skill_rows, correct_rows, stu_rows, opp_rows, data
    
    
    index = 0   
    tuple_rows = []
    while(index < len(rows)):
          if int(rows[index][0])>problem_len: 
                  
                  problems = int(rows[index][0]) 
                  student_id= int(rows[index][1])
                  studentids.append(student_id)  
                  
                                 
                  
                  
                  if (problems>problem_len):
                  
                  
                     tmp_max_steps = int(rows[index][0])
                     if(tmp_max_steps > max_steps):
                        max_steps = tmp_max_steps
                        
                        
                     asses= bkt_ass[student_id]
                                        
                     len_problems = int(int(problems) / problem_len)*problem_len
                     rest_problems = problems - len_problems             
                     
                     ele_p = []             
                     p_index=0       
                     for element in rows[index+1]:
                         ele_p.append(int(element))
                         p_index=p_index+1 
                         
                     ele_c = []
                     c_index=0
                     for element in rows[index+3]:
                         ele_c.append(int(element))
                         c_index=c_index+1
                         
                         
                     ele_d = []
                     d_index=0
                     for element in rows[index+2]:
                         ele_d.append(int(element))
                         d_index=d_index+1
                         
                         
                     ele_a = []             
                     a_index=0       
                     for element in asses:
                         ele_a.append(float(element))
                         a_index=a_index+1 

                     if (rest_problems>0):
                        rest=problem_len-rest_problems
                        for i in range(rest):
                            ele_p.append(-1)
                            ele_c.append(-1)
                            ele_d.append(-1)
                            ele_a.append(-1)

                     ele_p_array = np.reshape(np.asarray(ele_p), (-1,problem_len))
                     ele_c_array = np.reshape(np.asarray(ele_c), (-1,problem_len))
                     ele_d_array = np.reshape(np.asarray(ele_d), (-1,problem_len))
                     ele_a_array = np.reshape(np.asarray(ele_a), (-1,problem_len))
                   
                     n_pieces = ele_p_array.shape[0]
                     
                   
                     for j in range(n_pieces):
                         s1=[student_id,j,problems]
                         
                         if (j>-1) & (j< (n_pieces-1)) :
                            s1.append(1)
                            s2= np.append(ele_p_array[j,:],ele_p_array[j+1,0]).tolist()
                            s3= np.append(ele_c_array[j,:],ele_c_array[j+1,0]).tolist() 
                            s4= np.append(ele_d_array[j,:],ele_d_array[j+1,0]).tolist()  
                            s5= np.append(ele_a_array[j,:],ele_a_array[j+1,0]).tolist()      
                         else:
                              s1.append(-1)
                              s2= ele_p_array[j,:].tolist()
                              s3= ele_c_array[j,:].tolist() 
                              s4= ele_d_array[j,:].tolist() 
                              s5= ele_a_array[j,:].tolist() 
                         tup = (s1,s2,s3,s4,s5)
                         tuple_rows.append(tup)
          index += 4
          
          
          
    
    
    
    max_steps  =max_steps+1
    
    
    index=0
    train_students=[]
    test_students=[]            
    while(index < len(tuple_rows)):
         if (int(tuple_rows[index][0][0]) in train_ids):
            train_students.append(tuple_rows[index])
         if (int(tuple_rows[index][0][0]) in test_ids):
            test_students.append(tuple_rows[index])
         index+=1
    
      
    return train_students, test_students, studentids, max_skills, max_items, train_ids, test_ids
    
    
def get_bktdata(df): 
    BKT_dict = {}
    DKT_skill_dict = {}
    DKT_res_dict = {}   

    for kc in list(df['skills'].unique()):
        kc_df=df[df['skills']==kc].sort_values(['stus'],ascending=True)             
        stu_cfa_dict = {}
        
        for stu in list(kc_df['stus'].unique()):
            df_final=kc_df[kc_df['stus']==int(stu)].reset_index().sort_values(['opp'],ascending=True)
            stu_cfa_dict[int(stu)]=list(df_final['corrects'])
            
        BKT_dict[int(kc)]=stu_cfa_dict
        
        
    for stu in list(df['stus'].unique()):
        stu_df=df[df['stus']==int(stu)].sort_values(['opp'],ascending=True)
        DKT_skill_dict[int(stu)]=list(stu_df['skills'])
        DKT_res_dict[int(stu)]=list(stu_df['corrects'])
        
    

    return BKT_dict, DKT_skill_dict, DKT_res_dict
            
        

def cluster_data(students,max_stu,num_skills, datatype):

    
    

    success = []
    max_seg =0    
    xtotal = np.zeros((max_stu,num_skills))    
    x1 = np.zeros((max_stu,num_skills))
    x0 = np.zeros((max_stu,num_skills)) 
    
    index = 0  
    while(index+FLAGS.batch_size < len(students)):    
         for i in range(FLAGS.batch_size):
             student = students[index+i] 
             student_id = int(student[0][0])
             seg_id = int(student[0][1])
             
                 
             if (int(student[0][3])==1):
                tmp_seg = seg_id
                if(tmp_seg > max_seg):
                   max_seg = tmp_seg
                problem_ids = student[1]                
                correctness = student[2]
                for j in range(len(problem_ids)):           
                    key =problem_ids[j]
                    xtotal[student_id,key] +=1
                    if(int(correctness[j]) == 1):
                      x1[student_id,key] +=1
                    else:
                         x0[student_id,key] +=1

                xsr=[x/y  for x, y in zip(x1[student_id], xtotal[student_id])]
                
                x=np.nan_to_num(xsr)
                x=np.append(x, student_id)
                x=np.append(x, seg_id)
                success.append(x) 
                
                
         index += FLAGS.batch_size 
         
    return success, max_seg 
    
    
    
def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
    
    
def BKTAssessment(data, train_ids, max_skills):

    bkt_data, dkt_skill, dkt_res =get_bktdata(data)
    DL, DT, DG, DS ={}, {}, {}, {}
    for i in bkt_data.keys():
        skill_data = bkt_data[i]
        train_data=[]
        for j in skill_data.keys():
            if int(j) in train_ids:                       
               train_data.append(list(map(int,skill_data[j])))
                       
        bkt = BKT(step = 0.1, bounded = False, best_k0 = True)
        if len(train_data)>2:
           DL[i],DT[i],DG[i],DS[i]=bkt.fit(train_data)   
        else:
             DL[i],DT[i],DG[i],DS[i] = 0.5, 0.2, 0.1, 0.1   
        
    del bkt_data
    
    mastery =  bkt.inter_predict(dkt_skill, dkt_res, DL, DT, DG, DS, max_skills)
    
    del dkt_skill, dkt_res
    
    return mastery
    
    print("**************Finished BKT Assessment****************")



def main(unused_args):

    config = HyperParamsConfig()
    
    data_name= '4_Ass_09'
    cluster_num= FLAGS.num_cluster
    problem_len= FLAGS.problem_len
    
    
    train_data='./'+data_name+'_train.csv'
    test_data= './'+data_name+'_test.csv'
          
    train_students, test_students, student_ids, max_skills, max_items, train_ids, test_ids =read_data_from_csv_file(train_data, test_data)
    config.num_skills = max_skills
                 
    
    item_diff = difficulty_data(train_students+test_students,max_items)             

    train_cluster_data, train_max_seg= cluster_data(train_students,max(train_ids)+1,max_skills,"train")        
    test_cluster_data, test_max_seg= cluster_data(test_students,max(test_ids)+1,max_skills, "test")
    
    max_stu= max(student_ids)+1
    max_seg=max([int(train_max_seg),int(test_max_seg)])+1
            
            
    

            
    with tf.Graph().as_default():
         session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                       log_device_placement=FLAGS.log_device_placement)
         global_step = tf.Variable(0, name="global_step", trainable=False)
         starter_learning_rate = FLAGS.learning_rate
         learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)
         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
         with tf.Session(config=session_conf) as session:
              cluster =k_means_clust(session, train_cluster_data, test_cluster_data, max_stu, max_seg, FLAGS.num_cluster, max_skills, 40)
              del train_cluster_data, test_cluster_data
              initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
              # training model
              with tf.variable_scope("model", reuse=None, initializer=initializer):
                   m = StudentModel(is_training=True, config=config)
              # testing model
              with tf.variable_scope("model", reuse=True, initializer=initializer):
                   mtest = StudentModel(is_training=False, config=config)
              grads_and_vars = optimizer.compute_gradients(m.cost)
              grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
              for g, v in grads_and_vars if g is not None]
              grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
              train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
              session.run(tf.initialize_all_variables())
              j=1
              for i in range(config.max_max_epoch):
                  rmse, auc, r2, acc, _ = run_epoch(session, m, train_students, item_diff, max_stu, cluster, train_op,max_skills, "train", i,  verbose=False)
                  print(" Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2:  %.3f \t acc: %.3f \n" % ( i + 1, rmse, auc, r2, acc))
                  if((i+1) % FLAGS.evaluation_interval == 0):
                    rmse, auc, r2, acc, all_logits = run_epoch(session, mtest, test_students, item_diff, max_stu, cluster, tf.no_op(), max_skills, "test", j, verbose=True)
                    print(" Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \t acc: %.3f \n" % ( j, rmse, auc, r2, acc))

                    j+=1
                            
                            

                        

             
               
if __name__ == "__main__":
    tf.app.run()
