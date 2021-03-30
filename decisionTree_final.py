import numpy as np
import csv
import math

class DTree:
    def __init__(self,data,test_data,max_depth):
        root=Node(data,max_depth)

        self.root=root
        self.leftChild=root.leftChild
        self.rightChild=root.rightChild
        self.training_data = data
        self.test_data = test_data
        self.max_depth=max_depth
        #self.train_examples=train_examples
        examples = data[1:]
        self.examples=examples
        self.test_examples=test_data[1:]
        #depth=self.root.depth
    '''
    def output(self):
        self.root.pretty_print()
    '''
    def train(self):
        self.root.create_children()


    # Even Newer Predict

    def pred(self):
        row_data=self.training_data
        prediction_list=[]
        lab_list=[]
        for ex in row_data:
            prediction_list.append(self.root.get_label(ex,lab_list))   #,self.root.leftChild,self.root.rightChild))
        print(prediction_list)
        return prediction_list


    '''
    # New Predict
      def predict(self,data,self.root.depth, attr2split_names):
        p_labels = []
        attr2split_names = np.unique(data[1:, self.root.max_mi_col])
        print(attr2split_names)
        for ex in self.root.data[1:]:
            p_labels.append(self.predict(self.root, ex, self.root.depth, attr2split_names))
        print(p_labels)
        return self.predict(plabels)
    '''

'''
    def predict(self):
       no_of_examples=len(self.root.data[1:])
       mismatch=0
       for i in range(no_of_examples):
           if root.predicted_label(data[i]) != data[i,-1]
           mismatch+=1
       error=mismatch/no_of_examples
       return error
'''

def get_data(infile):
    with open(infile) as f:
        reader=csv.reader(f,delimiter='\t')
        data=[]
        for row in reader:
            data.append(row)
        npdata = np.array(data)
    return npdata

class Node():

    def __init__(self,data,max_depth,name=None,split_val=None,depth=0,leftChild=None,rightChild=None,majority=None,stats=None):
        #self.labeldata= labeldata
        self.data=data
        self.depth = depth
        self.max_depth=max_depth
        self.stats=stats
        self.name=name
        self.max_mi_col=None
        self.split_val=split_val
        #leftChild=self.create_children()
        #rightChild=self.create_children()
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.isLeaf= False
        self.isRoot = False
        #self.label= label
        self.majority_label= None
        self.attr2split_names=None
        self.children=[]
        #self.lc=self.create_children()[0]
        #self.rc=self.create_children()[1]
        #self.plusVal=plusval
        #self.minusVal=minusVal

        self.DTree =DTree

    # training functions

    # Return format : attribute col index, name,  max MI
    def Mutual_Information(self, data):
        # Majority Label
        '''def max_label_or_val(npdata, index):
            max_lab_or_val = mode(npdata[1:, index])[0][0]
            return max_lab_or_val'''

        def label_and_vals(npdata, attr_col_no):
            uniq_label_names, uniq_label_counts = np.unique(npdata[1:, -1], return_counts=True)
            attr2split_names, attr2split_val_counts = np.unique(npdata[1:, attr_col_no], return_counts=True)
            # above line o/p format: ['n' 'y'] [14 14]
            return uniq_label_names, uniq_label_counts, attr2split_names, attr2split_val_counts

        def entropy(uniq_label_counts, uniq_label_names):
            entropy = 0
            for i in range(len(uniq_label_names)):
                entropy += ((uniq_label_counts[i] / (np.sum(uniq_label_counts)) * math.log(
                    uniq_label_counts[i] / np.sum(uniq_label_counts), 2)))
                # + (label_counts[1]/(np.sum(label_counts))*math.log(label_counts[1]/np.sum(label_counts),2)))
            return -entropy

        def cond_data(npdata, cond_attr, cond_val):  # Data when cond_attr has value 'cond_val'
            cond_data = npdata[npdata[:, cond_attr] == cond_val]
            return cond_data

        def cond_entropy(npdata, attr_col_no, attr2split_names, attr2split_val_counts):
            cond_entropy = 0
            for n, j in zip(attr2split_names, range(len(attr2split_names))):
                cond_data = npdata[npdata[:, attr_col_no] == n]
                uniq_label_names, uniq_label_counts = np.unique(cond_data[1:, -1], return_counts=True)
                for i in range(len(uniq_label_names)):
                    cond_entropy += (attr2split_val_counts[j] / np.sum(attr2split_val_counts)) * (-(
                            uniq_label_counts[i] / (np.sum(uniq_label_counts)) * math.log(
                        uniq_label_counts[i] / np.sum(uniq_label_counts), 2)))
            return cond_entropy

        def MI(entropy, cond_entropy):
            return entropy - cond_entropy

        npdata = data
        cols = np.size(npdata[:, :-1], 1) #number of columns

        if cols==0:
            self.isLeaf=True
            return
        MI_matrix = []

        for c in range(cols):
            attr_col_no = c
            uniq_label_names, uniq_label_counts, attr2split_names, attr2split_val_counts = label_and_vals(npdata, attr_col_no)
            ent = entropy(uniq_label_counts, uniq_label_names)
            con_ent = cond_entropy(npdata, attr_col_no, attr2split_names, attr2split_val_counts)
            n=npdata[0,c]
            MI_matrix.append([c,n, MI(ent, con_ent)])
        #print(MI_matrix)
        # Gets attribute with max mutual information
        def max_MI_attr (MI_matrix):
            max_MI=max(MI_matrix, key=lambda x: x[2])
            return max_MI

        max_MI = max_MI_attr(MI_matrix)
        self.max_mi_col=max_MI[0]

        # return max_MI[0], npdata[0,max_MI], max_MI[1] # return max MI attribute col index, name and max MIt_va
        return max_MI

    # Returns arrays of data matrices in dict with keys as vals

    # Returns arrays of leftData and rightData
    '''
     def suff_stats(self):
        label_names, label_counts = np.unique(self.data[:, -1], return_counts=True)
        zipped = zip(label_names, label_counts)
        lst_zip = list(zipped)
        # [ 'rep' ,'dem'] [12,  1]
        if len(lst_zip) == 2:
            p_print_suff_stats = '[' + str(lst_zip[0][1]) + " " + str(lst_zip[0][0]) + "/" + str(
                lst_zip[1][1]) + " " + str(lst_zip[1][0]) + ']'
        else:
            p_print_suff_stats="short list: " + str(lst_zip)
            #p_print_suff_stats = '[' + str(lst_zip[0][1]) + " " + str(lst_zip[1][0]) + ']'
        # self.stats=p_print_suff_stats   # Try and include this if there is suff stats realated error
        return p_print_suff_stats
    '''

    def split_data(self,data,attr2split,attr2split_names):
        left_data = data[data[:,attr2split] == attr2split_names[0]]
        right_data = data[data[:, attr2split] == attr2split_names[1]]

        return left_data,right_data



    def create_children(self):

        # Getting Max MI attribute column no and name
        max_mi_col = self.Mutual_Information(self.data)[0]  # column index of Max MI attribute
        self.name = self.Mutual_Information(data)[1]  # name of Max MI attribute
        self.max_mi_col=max_mi_col

        # Storing Split attribute names
        attr2split_names = np.unique(self.data[1:, max_mi_col])
        self. attr2split_names=attr2split_names
        print(attr2split_names)

        # to check if leaf by: Checking if no more examples, Chemax_mi_col = self.Mutual_Information(self.data)[0]  # column index of Max MI attribute
        # max_mi_col: attribute of this branch's splitcking if no more attributes, Checking if MI is positive

        #print("data:",self.data)
        #print(self.data)
        #print("Depth:",self.depth)
        #print("Max attr,MI:",self.max_mi_col, self.Mutual_Information(self.data)[2])

        if (len(self.data[1:]) == 0) or (len(self.data[0])-1 == 0) or (self.Mutual_Information(self.data)[2] <= 0) or (self.depth==self.max_depth):
            print("Is Leaf")
            a = self.data[:,-1]
            (d, counts) = np.unique(a, return_counts=True)

            index=np.argmax(counts)
            majority_label=a[index]
            #print(majority_label)
            self.majority_label=majority_label
            self.isLeaf=True
            #print("Max_MI_Col:",self.max_mi_col)
            return None, None    # Return something that says this is a leaf node and no more children will be generated

        else:
            #Condition for non-binary tree
            if len(attr2split_names) > 2:
                print("Not a Binary Tree")
                return None

            # Splitting the Data ino LeftData and RightData
            left_data = self.data[self.data[:, max_mi_col] == attr2split_names[0]]
            #print("Leftdata:", left_data)

            #print(np.shape(left_data)[1])
            if np.shape(left_data)[1]>2:
                left_data=np.delete(left_data,max_mi_col,1)             # Removing the max attribute column
            # print(left_data)

            if len(attr2split_names)==2:
                right_data = self.data[self.data[:, max_mi_col] == attr2split_names[1]]
            elif len(attr2split_names)==1:
                right_data = self.data[self.data[:, max_mi_col] == attr2split_names[0]]

            #print("right data:", right_data)
            # print(np.shape(right_data)[1])

            if np.shape(right_data)[1] > 2:
                right_data=np.delete(right_data, max_mi_col, 1)        # Removing the max attribute column

            depth = self.depth+1
            #print("Depth:",depth)

            self.split_val = attr2split_names[0]
            #print("left split val",self.split_val)

            leftChild= Node(left_data,max_depth=md,name=self.name,split_val=attr2split_names[0],depth=depth) # get md from terminal
            leftChild.max_mi_col=leftChild.Mutual_Information(leftChild.data)[0]
            a = leftChild.data[:, -1]
            (d, counts) = np.unique(a, return_counts=True)
            # print("counts:",d,counts)
            index = np.argmax(counts)
            majority_label = a[index]
            # print(majority_label)
            self.majority_label = majority_label
            #print("LMaj,Max_attr:", leftChild.majority_label,self.max_mi_col)

            #print("Max_MI_Col:", leftChild.max_mi_col)
            #print("LeftChildData:",leftChild.data)
            self.leftChild=leftChild.create_children()
            self.children.append(leftChild)


            if len(attr2split_names) == 2:
                self.split_val = attr2split_names[1]



            elif len(attr2split_names)==1:
                self.split_val = attr2split_names[0]
            #print("right split val", self.split_val)

            if len(attr2split_names) == 2:
                rightChild= Node(right_data,max_depth=md,name=self.name,split_val=attr2split_names[1]) # get md from terminal
            elif len(attr2split_names) == 1:
                rightChild = Node(right_data, max_depth=md, name=self.name, split_val=attr2split_names[0])



            rightChild.max_mi_col = rightChild.Mutual_Information(rightChild.data)[0]
            a = rightChild.data[:, -1]
            (d, counts) = np.unique(a, return_counts=True)
            # print("counts:",d,counts)
            index = np.argmax(counts)
            majority_label = a[index]
            # print(majority_label)
            self.majority_label = majority_label
            #print("RMaj,Max_attrr:",rightChild.majority_label,self.max_mi_col)
            #print("Max_MI_Col:", leftChild.max_mi_col)
            #print("RightChildData:", rightChild.data)
            self.rightChild=rightChild.create_children()
            self.children.append(rightChild)
        #for e in self.children:
            #print("Children data:",e.data)
        return leftChild, rightChild


    # Other functions
    '''
    def prediction(self,node_data):
    return classified_label

    def get_attr_counts(self, data,attr_col_index):
        npdata = np.array(data)
        attributes = npdata[1:, :-1]
        attributes_tx = np.transpose(attributes)
        attr_names = npdata[0, :-1]
        attr_values, attr_counts = np.unique(attributes_tx[attr_col_index], return_counts=True)
        print(attr_values, attr_counts)
        return attr_values,attr_counts

    '''


    '''
        def pretty_print(self):
        if self.leftChild == None and self.rightChild == None:  # and self.depth==0
            print('This')
            print(self.stats)
        else:
            print('Or this')
            print(self.name)
            print(self.stats)
            string="| "*self.depth + self.name + " = " + self.split_val + ": " + self.stats
            print(string)
        if self.leftChild:
            self.pretty_print()
        if self.rightChild:
            self.pretty_print()
    '''

# Mutual_Information(data) gives MI of all attributes in form [[0, 0.6050729559297063], [1, 0.2747983642304346]]

    def get_label(self,example,lab_list):    # ,lab_list  # example should be a single example without headers

         if self.isLeaf==True:   #self.leftChild==None and self.rightChild==None
             return self.majority_label
         for e in self.children:
             if example[e.max_mi_col]==e.attr2split_names[0]:
                lab_list.append(e.majority_label)
             if example[e.max_mi_col]==e.attr2split_names[1]:
                lab_list.append(e.majority_label)


'''if self.isLeaf == True:  # self.leftChild==None and self.rightChild==None
    return self.majority_label'''
'''
        if self.leftChild == None or self.rightChild == None:

    return self.majority_labels

if example[self.max_mi_col]==self.attr2split_names[0]:
    self.get_label(example)

if example[self.max_mi_col] == self.attr2split_names[1]:
    self.get_label(example)
'''



'''
    def predict(self,root,example,depth):
        print(self.root.attr2split_names)
        if depth > self.max_depth:
            return None
        if example[root.max_mi_col]==self.root.attr2split_names[0]:
            self.predict(root.leftChild,example, depth+1,self.root.attr2split_names)
        else:
            #example[root.max_mi_col]==attr2split_names[1]
            self.predict(root.rightChild,example, depth+1,self.root.attr2split_names)
'''
if __name__=="__main__":

    training_infile = 'handout/politicians_test.tsv'
    test_infile = "handout/small_test.tsv"
    outfile = "metrics out.txt"

    # Returns np Data array (including labels and headers)
    def get_data(infile):
        with open(infile) as f:
            reader = csv.reader(f, delimiter='\t')
            data = []
            for row in reader:
                data.append(row)
                npdata = np.array(data)
        return npdata

        # Returns


    data = get_data(training_infile)
    examples = data[1:]
    test_data = get_data(test_infile)
    md = 4

    # get from terminal
    # Calculation of output metrics
    # repeat below for training and test

    DT = DTree(data,test_data,max_depth=md)
    DT.train()
    # DT.pred()
    # DT.predict(test_data)

    # root.create_children(data)
    # root.pretty_print()


    '''
    #write output
    output="error(train): " + error(data)+ '\n'+  "error(train): " + error(test_data)
    with open(outfile, 'w') as out_f:
        out_f.write(output)
    '''
