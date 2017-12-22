import pandas as pd
import math
	
def entropy_of_1(m1, n1):
	"""Returns the the entropy of the parameters passed.

	Args:
    	m1:	Number of occurences of a value of the attribute
        n1:	Number of occurences of the other value of attribute

    Returns:
    	Returns the the entropy of the parameters passed .
	"""
	if m1 == 0 and n1 == 0:
		return 0
	a = m1 * 1.0/(m1 + n1)
	b = n1 * 1.0/(m1 + n1)
	# To deal with log of 0 to the base 2 = 0 condition.
	if a == 0.0 and b == 0.0:
		return 0
	elif a == 0.0:
		return -(b * math.log(b,2))
	elif b == 0.0:
		return -(a * math.log(a,2))
	else:
		return (-(a*math.log(a,2))-(b*math.log(b,2)))
	
def entropy_of_0(m2, n2):
	"""Returns the the entropy of an instance where attribute value is 0.

	Args:
    	m2:	Number of cases where value of attribute is 0 and that of class is also 1 for a particular instance
        n2:	Number of cases where value of attribute is 0 and that of class is also 0 for a particular instance

    Returns:
        Calls function entropy_of_1 to calculate the entropy
	"""
	return entropy_of_1(m2, n2)	

	
def entropy_of_parent(target_attribute_df):
	"""Returns the the entropy of an instance where attribute value is 0.

	Args:
    	target_attribute_df(a dataframe): This contains the class column of given dataframe

    Returns:
        Calls function entropy_of_1 to calculate the entropy

	"""
	class_0, class_1 = 0, 0
	class_1 = (target_attribute_df['Class'] == 1).sum() # Calculate the number of occurences of '1' in the class column
	class_0 = (target_attribute_df['Class'] == 0).sum() # Calculate the number of occurences of '0' in the class column
	return entropy_of_1(class_1, class_0)
	
	
def IG(current_node_df, target_attribute_df):
	"""
	Returns the Information Gain of the attribute column passed by using the class column passed in target_attribute_df
	
	Args:
        current_node_df(a dataframe): This contains one attribute column of the entire dataframe.
        target_attribute_df(a dataframe): This contains the class column of given dataframe

    Returns:
        Returns the the calculated Information Gain
        IG = Entropy Of parent - weighted average of entropy of all children

    Variable denotions:

    m1:	Number of cases where value of attribute is 1 and that of class is also 1 for a particular instance
    n1:	Number of cases where value of attribute is 1 and that of class is also 0 for a particular instance
    m2:	Number of cases where value of attribute is 0 and that of class is also 1 for a particular instance
    n2:	Number of cases where value of attribute is 0 and that of class is also 0 for a particular instance

	"""
	m1, n1, m2, n2 = 0, 0, 0, 0
	entropy_parent = entropy_of_parent(target_attribute_df)
	# Iterates over the current_node_df for calculating the total count of all the cases for an attribute
	for j, row in current_node_df.iterrows():
		if(current_node_df.iloc[:, 0][j] == 1 and target_attribute_df.iloc[:, 0][j] == 1):		
			m1 = m1 + 1
		if(current_node_df.iloc[:, 0][j] == 1 and target_attribute_df.iloc[:, 0][j] == 0):
			n1 = n1 + 1	
		if(current_node_df.iloc[:, 0][j] == 0 and target_attribute_df.iloc[:, 0][j] == 1):
			m2 = m2 + 1	
		if(current_node_df.iloc[:, 0][j] == 0 and target_attribute_df.iloc[:, 0][j] == 0):
			n2 = n2 + 1	
	total_count = m1 + n1 + m2 + n2  #Store the total count of all the cases for an attribute
	entropy_1 = entropy_of_1(m1, n1)
	entropy_0 = entropy_of_0(m2, n2)
	# Calculate the weighted average of the entropies of children
	weighted_entropy_children = (((m1 + n1) * 1.0/total_count) * entropy_1) + (((m2 + n2) * 1.0/total_count) * entropy_0) # Calculate the weighted average of the entropies of children
	return (entropy_parent - weighted_entropy_children)
	

def find_best(df, target_attribute_df, attributes_df, attribute_no, depth):
	"""
	Returns the attribute to best classify instances among all the given attributes by comparing thier Information Gain Values
	
	Args:
        df(dataframe): This is a dataframe containing all the attributes among which best needs to be found
        				 and the class for all instances as well
        target_attribute_df(dataframe): This contains the class column of given dataframe
        attributes_df(dataframe): It contains only the attributes and excludes the class column
        attribute_no(Integer): An Integer to select a range of attributes among a given dataframe.
        depth(Integer): Stores the depth of the tree

    Returns:
        Returns the best attribute among all the given attributes

   	global Variables:
   		dt_nodes(List): To store the calculated best attribute among the available attributes
   		IG_list_index(Integer): Index of the best attribute i.e. with maximum IG 
   		index_dt_nodes(Integer): Index of an attribute in dt_nodes
   		root: The root node of the tree

	"""

	global IG_list_index, index_dt_nodes
	if depth != 0:
		df = df.drop(dt_nodes, axis = 1)		#	Drop the column of attributes at the last position in dt_nodes
	IG_list = []	#	List to store IG Values of all the instances of an attribute.
	new_attributes_df = df.iloc[:, :(len(df.columns) - 1)]
	current_node_df = new_attributes_df.iloc[:, (attribute_no - 1):attribute_no]
	new_target_attribute_df = df.loc[:, ['Class']]
	# Store the Information Gain values for all the attributes in the IG_list
	if(len(dt_nodes) != (len(dataframe.columns)) - 1):
		for x in attributes_df.iteritems():
			IG_list.append(IG(current_node_df, new_target_attribute_df))
			if attribute_no < (len(df.columns) - 1):
				current_node_df = new_attributes_df.iloc[:, attribute_no: (attribute_no + 1)]
				attribute_no = attribute_no + 1
		IG_list_index = IG_list.index(max(IG_list))		# 'IG_list_index' is the index of the attribute i.e. with maximum IG 
		dt_nodes.append(attributes_df.columns[IG_list_index])
		global root
		root = dt_nodes[0]	#	Stores the root node of the decision tree
		index_dt_nodes = len(dt_nodes) - 1
		new_attributes_df = attributes_df.iloc[:,:] #	Make a copy of attributes_df
		#	Remove the attribute column from the new_attributes_df which has been selected as the best attribute for classification
		del new_attributes_df[new_attributes_df.columns[IG_list_index]] 
	
	
def classify(depth):
	"""Classifies the entire dataframe for a particular attribute value( '1' or '0')

	Args:
    	depth(Integer): Stores the depth of the tree
	"""
	dataframe_1 = dataframe.loc[dataframe[dt_nodes[temp_index_dt_nodes]] == 1]  # Classify the dataframe for attribute vaue '1'
	find_best(dataframe_1, target_attribute_df, new_attributes_df, 1, depth)    # Call find best to find the best attribute in the classified dataframe
	dataframe_0 = dataframe.loc[dataframe[dt_nodes[temp_index_dt_nodes]] == 0]  # Classify the dataframe for attribute vaue '0'
	find_best(dataframe_0, target_attribute_df, new_attributes_df, 1, depth)	# Call find best to find the best attribute in the classified dataframe
	
	
def ID3(df, target_attribute_df, attributes_df, depth, x):
	"""Recursive function to implement the ID3 decision tree recursively.

	Args:
        df(dataframe): This is a dataframe containing all the attributes among which best needs to be found
        				 and the class for all instances as well
        target_attribute_df(dataframe): This contains the class column of given dataframe
        attributes_df(dataframe): It contains only the attributes and excludes the class column
        depth(Integer): Stores the depth of the tree
        x(Integer): Used to Update values of temp_index_dt_nodes

    global Variables:
    	temp_index_dt_nodes(Integer): Stores the index of dt_nodes List temporarily
	"""
	global temp_index_dt_nodes
	if depth == 0:
		temp_index_dt_nodes = 0
	for j, row in target_attribute_df.iterrows():
		if(target_attribute_df.iloc[:, 0][j] == 0):
			flag_0 = 1
	if (flag_0 != 1):
		# If all examples are positive, Return the single-node tree Root, with label = positive.
		print 'positive'
		return
	for j, row in target_attribute_df.iterrows():
		if(target_attribute_df.iloc[:, 0][j] == 1):
			flag_1 = 1
	if (flag_1 != 1):
		# If all examples are negative, Return the single-node tree Root, with label = negative.
		print 'negative'
		return
	if new_attributes_df.empty:
		# If number of predicting attributes is empty, then Return the single node tree Root,
    	# with label = most common value of the target attribute in the examples.
		if len(dataframe) != 1:
			return
		count_0 = 0
		count_1 = 0
		for j, row in target_attribute_df.iterrows():
			if(target_attribute_df.iloc[:, 0][j] == 1):
				count_1 = count_1 + 1
		for j, row in target_attribute_df.iterrows():
			if(target_attribute_df.iloc[:, 0][j] == 0):
				count_0 = count_0 + 1	
		if count_1 >= count_0:
			print root, ': 1' 
			return
		else:
			print root, ': 0'
			return
	#	For Root node
	if depth == 0:
		find_best(df, target_attribute_df, attributes_df, 1, depth)
		depth = depth + 1 		# Update Depth
		return ID3(df, target_attribute_df, new_attributes_df, depth, x)
	# 	For all other nodes including leaf nodes.
	else:
		classify(depth)
		temp_index_dt_nodes = index_dt_nodes - x
		x = x + 1
		return ID3(dataframe, target_attribute_df, new_attributes_df, depth, x) #	Call ID3 recursively
		
		
def create_2d_list(dt_nodes):
	"""Creates a two dimensional list containing attributes which are
	   split according to attribute value '0' and attribute value '1' of their parent attribute
	   in the index '0' and '1' of (the inner list of )create_2d_list respectively.
	   The index of outer list of the create_2d_list corresponds to that of dt_nodes.

	Args:
    	dt_nodes(List): The list of attributes in which attribute names are appended in 
    					an order as the attribute to best classify instances is calculated
    					for the available dataframe.

    Returns:
    	Returns the created two dimensional list of attributes
	"""
	i = 0
	list_2d = []
	while i < (len(dt_nodes)/2):
		list_2d.append([])
		i = i + 1
	i = 0	
	x = i + 1
	# Add attribute names from dt_nodes to list_2d
	while i < (len(dt_nodes)/2):
		list_2d[i].append(dt_nodes[i + x])
		if i != (len(dt_nodes)/2) - 1:
			list_2d[i].append(dt_nodes[i + (x + 1)])
		x = x + 1
		i = i + 1
	return list_2d
	

def leaf_node(temp_current_element, temp_values):
	"""Returns the value of the Leaf node of the tree using the attributes 
	   in the temp_current_element list and Values of those attributes in 
	   the temp_values list

	Args:
    	temp_current_element(List): Stores the list of attributes required for finding the value of the leaf node
    	temp_values(List): Stores the list of values of attributes required for finding the value of the leaf node

	"""
	count_0 = 0
 	count_1 = 0 
 	x = 0
 	i = 0
	global correct_count # To count the number of correctly classified instances 
 	new_dataframe = dataframe
 	while i<len(temp_values):
 		# If temp_value is '0', filter the new_training_df according to attribute in temp_current_element list having value '0'
 		if(temp_values[i] == 0):
			new_dataframe = new_dataframe.loc[new_dataframe[temp_current_element[x]] == 0]
			x = x + 1
			i = i + 1
		# If temp_value is '1', filter the new_training_df according to attribute in temp_current_element list having value '1'
		elif(temp_values[i] == 1):
			new_dataframe = new_dataframe.loc[new_dataframe[temp_current_element[x]] == 1]
			x = x + 1
			i = i + 1
	value_1 = (new_dataframe['Class'] == 1).sum() # Calculate the number of occurences of '1' in the class column
	value_0 = (new_dataframe['Class'] == 0).sum() # Calculate the number of occurences of '0' in the class column
	# If number of occurences of '1' in the class column is greater than that of zero return '1' else return '0'
	# Also update count of correctly classified instances 
	if value_1 >= value_0:
		correct_count = correct_count + value_1
		return 1
	else:
		correct_count = correct_count + value_0
		return 0

def print_dt_1(list_2d):
	"""Prints the half of the decision tree where the root element has value '1'

	Args:
		list_2d(List): a two dimensional list containing attributes which are split according to attribute value '0' 
		and attribute value '1' of their parent attribute in the index '0' and '1' of (the inner list of )create_2d_list 
		respectively.
	    The index of outer list of the create_2d_list corresponds to that of dt_nodes.
    
    """
	global count_leaf_node
	tab_count = 1 				# For adjusting spaces while printing the tree.
	print dt_nodes[0], " = 1:"
	current_element = []		# List which stores the attributes until which the tree has traversed 
	temp_current_element = []	# List which stores the attributes until which the tree has traversed temporarily
	temp_values = []			# List which stores the values of the attributes till the leaf node
	temp_current_element.append(dt_nodes[0])
	temp_values.append(1)
	previous_element = [2]      # List to keep track of the previous element that was printed in the tree
	x = 2 
	n = 1
	current_element.append(list_2d[0][0])
	temp_current_element.append(list_2d[0][0])
	temp_values.append(1)
	#	This loop iterates over all the attributes in the dt_list list and prints them in the form of a tree.
	while (previous_element[len(previous_element) - 1] != dt_nodes[0]): 
		# To initialise the previous element list
		if (previous_element[len(previous_element) - 1] == 2):
			previous_element = []
		if (dt_nodes.index(current_element[len(current_element) - 1]) <= len(list_2d) - 1):
			print "|\t" * tab_count, current_element[len(current_element) - 1], "= 1:"
			tab_count = tab_count + 1
			previous_element.append(current_element[len(current_element) - 1])
			current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][0]) 
			temp_current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][0])
			temp_values.append(1)
		else:
			# Call the leaf_node function to print the value of the leaf node.
			print "|\t" * tab_count, current_element[len(current_element) - 1], "= 1:", leaf_node(temp_current_element, temp_values)
			#Update temp_values
			del temp_values[-1]
			temp_values.append(0)
			print "|\t" * tab_count, current_element[len(current_element) - 1], "= 0:", leaf_node(temp_current_element, temp_values)
			tab_count = tab_count - 1
			count_leaf_node = count_leaf_node + 1
			temp_current_element.remove(temp_current_element[len(temp_current_element) - 1]) #remove last element from temp
			del temp_values[-1]
			if (previous_element[len(previous_element) - 1] == dt_nodes[(len(dt_nodes) / 2) - 1] and current_element[len(current_element) - 1] == dt_nodes[len(dt_nodes) / 2]):
				return				
			elif (previous_element[len(previous_element) - 1] == dt_nodes[(len(dt_nodes) / 2) - 1]):
				del temp_values[-1]
				temp_values.append(0)
				print "|\t" * tab_count, previous_element[len(previous_element) - 1], "= 0:", leaf_node(temp_current_element, temp_values)
				count_leaf_node = count_leaf_node + 1
				temp_current_element.remove(temp_current_element[len(temp_current_element) - 1])
				del temp_values[-1]
			else:
				print "|\t" * tab_count, previous_element[len(previous_element) - 1], "= 0:" 
				del temp_values[-1]
				temp_values.append(0)
			tab_count = tab_count + 1	
			try:
				current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][1])
				temp_current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][1])
				temp_values.append(1)
				print "|\t" * tab_count, current_element[len(current_element) - 1], "= 1:", leaf_node(temp_current_element, temp_values)
				#Update temp_values
				del temp_values[-1]
				temp_values.append(0)
				print "|\t" * tab_count, current_element[len(current_element) - 1], "= 0:", leaf_node(temp_current_element, temp_values)
				tab_count = tab_count - 1
				count_leaf_node = count_leaf_node + 1
				temp_current_element.remove(temp_current_element[len(temp_current_element) - 1])
				del temp_values[-1]
				i = 1

				while i <= n:
					temp_current_element.remove(temp_current_element[len(temp_current_element) - 1])
					del temp_values[-1]
					i = i + 1
				n = n + 1
				if (current_element[len(current_element) - 1] == dt_nodes[len(dt_nodes) / 2]):
					previous_element.append(dt_nodes[0])	
				elif (x == 5):	
					tab_count = tab_count - 1
					temp = previous_element[len(previous_element) - x + 3] 
					#Update temp_values when backtracking to previous depth
					print "|\t" * tab_count, temp, "= 0:"
					del temp_values[-1]
					temp_values.append(0)
					tab_count = tab_count + 1	
				else:	
					tab_count = tab_count - n
					tab_count = tab_count + 1
					temp = previous_element[previous_element.index(temp) - 1] 
					print "|\t" * tab_count, temp, "= 0:"
					#Update temp_values when backtracking to previous depth
					del temp_values[-1]
					temp_values.append(0)
					tab_count = tab_count + 1
			# To handle IndexError when leaf node is not found
			except IndexError:
				tab_count = tab_count - 2
				temp = previous_element[len(previous_element) - 2]
				print "|\t" * tab_count, temp, "= 0:"
				del temp_values[-1]
				temp_values.append(0)
				tab_count = tab_count + 1	
			current_element.append(list_2d[dt_nodes.index(temp)][1])
			temp_current_element.append(list_2d[dt_nodes.index(temp)][1])
			temp_values.append(1)
		x = x + 1


def print_dt_2(list_2d):
	global count_leaf_node
	tab_count = 1 	# For adjusting spaces while printing the tree.
	print dt_nodes[0], " = 0:"
	current_element = []		# List which stores the attributes until which the tree has traversed 
	temp_current_element = []	# List which stores the attributes until which the tree has traversed temporarily
	temp_values = []			# List which stores the values of the attributes till the leaf node
	temp_current_element.append(dt_nodes[0])
	temp_values.append(0)
	previous_element = [2]
	x = 2
	n = 1
	current_element.append(list_2d[0][1])
	temp_current_element.append(list_2d[0][1])
	temp_values.append(1)
	#This loop iterates over all the attributes in the dt_list list and prints them in the form of a tree.
	while (previous_element[len(previous_element) - 1] != dt_nodes[0]):
		if (previous_element[len(previous_element) - 1] == 2):
			previous_element = []
		if (dt_nodes.index(current_element[len(current_element) - 1]) <= len(list_2d) - 1): 										
			print "|\t" * tab_count, current_element[len(current_element) - 1], "= 1:"
			tab_count = tab_count + 1
			previous_element.append(current_element[len(current_element) - 1])
			current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][0])  
			temp_current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][0])
			temp_values.append(1)

		else:
			print "|\t" * tab_count, current_element[len(current_element) - 1], "= 1:", leaf_node(temp_current_element, temp_values)
			del temp_values[-1]
			temp_values.append(0)
			print "|\t" * tab_count, current_element[len(current_element) - 1], "= 0:", leaf_node(temp_current_element, temp_values)
			tab_count = tab_count - 1
			count_leaf_node = count_leaf_node + 1
			temp_current_element.remove(temp_current_element[len(temp_current_element) - 1])
			del temp_values[-1]
			print "|\t" * tab_count, previous_element[len(previous_element) - 1], "= 0:"
			del temp_values[-1]
			temp_values.append(0)
			tab_count = tab_count + 1	
			try:
				current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][1])
				temp_current_element.append(list_2d[dt_nodes.index(previous_element[len(previous_element) - 1])][1])
				temp_values.append(1)
				print "|\t" * tab_count, current_element[len(current_element) - 1], "= 1:", leaf_node(temp_current_element, temp_values)
				del temp_values[-1]
				temp_values.append(0)
				print "|\t" * tab_count, current_element[len(current_element) - 1], "= 0:", leaf_node(temp_current_element, temp_values)
				count_leaf_node = count_leaf_node + 1
				temp_current_element.remove(temp_current_element[len(temp_current_element) - 1])
				del temp_values[-1]
				temp_current_element.remove(temp_current_element[len(temp_current_element) - 1])
				del temp_values[-1]
				if (previous_element[len(previous_element) - 1] == dt_nodes[6] and current_element[len(current_element) - 1] == dt_nodes[14]):
			 		return
				tab_count = tab_count - 1
				if (x == 4):
					tab_count = tab_count - 1
					temp = previous_element[len(previous_element) - x + 2]
					print "|\t" * tab_count, temp, "= 0:"
					del temp_values[-1]
					temp_values.append(0)
					tab_count = tab_count + 1
				else:
					tab_count = tab_count - 2
					temp = previous_element[previous_element.index(temp) - 1]
					print "|\t" * tab_count, temp, "= 0:"
					del temp_values[-1]
					temp_values.append(0)
					tab_count = tab_count + 1
			# To handle IndexError when leaf node is not found
			except IndexError:
				tab_count = tab_count - 2
				temp = previous_element[len(previous_element) - 2]
				print "|\t" * tab_count, temp, "= 0:"
				del temp_values[-1]
				temp_values.append(0)
				tab_count = tab_count + 1
			current_element.append(list_2d[dt_nodes.index(temp)][1])
			temp_current_element.append(list_2d[dt_nodes.index(temp)][1])
			temp_values.append(1)
		x = x + 1
		
global dt_nodes, correct_count, count_leaf_node

# TRAINING SET #
correct_count = 0	
dt_nodes = []
count_leaf_node = 0
location = raw_input("Please enter the training dataset path:")
file = open(location,'r')
dataframe = pd.read_csv(location, header = 0)
target_attribute_df = dataframe.loc[:, ['Class']]
attributes_df = dataframe.iloc[:, :(len(dataframe.columns) - 1)]
new_attributes_df = attributes_df.iloc[:,:]
		
ID3(dataframe, target_attribute_df, attributes_df, 0, 1)
list_2d = create_2d_list(dt_nodes)
print_dt_1(list_2d)
print_dt_2(list_2d)
count_training_instances = dataframe.shape[0]
count_training_attributes = dataframe.shape[1] - 1
count_tree_nodes_training = len(dt_nodes)
accuracy_training = (correct_count * 1.0 / count_training_instances)* 100

print "\n\n\n"
print "Pre-Pruned Accuracy"
print "-" * 20
print ""
print "Number of training instances = ", count_training_instances
print "Number of training attributes = ", count_training_attributes 
print "Total number of nodes in the tree = ", count_tree_nodes_training
print "Number of leaf nodes in the tree = ", count_leaf_node
print "Accuracy of the model on the training dataset = ", accuracy_training, "%"




	



