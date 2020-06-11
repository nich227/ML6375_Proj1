'''
Name:   Kevin Chen
Professor:  Haim Schweitzer
Due Date:   
Project 1
Python 3.7.6
'''

import re
import sys
import math
from collections import Counter

# Reads from an input dataset and an input partition


def read_input_files(dataset_file, partition_file):
    # Process dataset
    input_dataset = []

    with open(dataset_file) as dataset:
        dataset_line = dataset.readline()

        # Read first line; m, number of instances and n, number of features
        if not re.compile('\d+\s\d+').match(dataset_line):
            raise Exception(
                'ERROR: Dataset input file is in an unrecognized format. See the example input file for the correct format.')

        num_instances, num_features = (
            int(dataset_line.split()[0]), int(dataset_line.split()[1]))

        # Read in dataset
        for i in range(0, num_instances):
            dataset_line = dataset.readline()
            if not re.compile('(\d+\s+)+\d+').match(dataset_line):
                raise Exception(
                    'ERROR: Dataset input file is in an unrecognized format. See the example input file for the correct format.')

            if len(dataset_line.split()) != num_features:
                raise Exception(
                    'ERROR: A line does not have the correct amount of features.')

            input_dataset.append(list(map(int, dataset_line.split())))

    # Process partition
    input_partitions = {}

    with open(partition_file) as partition:
        partition_line = partition.readline()

        while partition_line:
            if not re.compile('\w+\s+(\d+\s+)+\d+').match(partition_line):
                raise Exception(
                    'ERROR: Partition input file is in an unrecognized format. See the example input file for the correct format.')

            input_partitions[partition_line.split()[0]] = list(map(int, partition_line.split()[
                1:]))
            partition_line = partition.readline()

    return input_dataset, input_partitions


# Calculate entropy of a partition
def calculate_entropy(partition, target_attr):

    partition_target_attr = []
    # Get only target attributes within the partition
    for index in partition:
        partition_target_attr.append(target_attr[index-1])

    # Count number of instances within the partition of unique target attributes
    category_counts = Counter(partition_target_attr).values()

    # Calculate entropy
    entropy = 0
    for count in category_counts:
        entropy += (count / len(partition)) * math.log2(len(partition) / count)

    return entropy


# Calculate entropy of a partition given a specific attribute
def calculate_entropy_attr(partition, this_attr, target_attr):

    # Get all unique categories for this attributes
    this_categories = list(Counter(this_attr).keys())

    # Count number of instances within the partition of unique this attributes
    this_category_counts = [0] * len(this_categories)
    for index in partition:
        this_category_counts[this_categories.index(this_attr[index-1])] += 1

    # Get entropy of each target attribute for each unique this attribute
    mini_partition_this_category = []
    this_target_attr = []
    this_category_index = 0

    total_entropy = 0
    for this_category in this_categories:

        # Narrow it down to this attribute in terms of partition indices and target attributes, to calculate entropy
        for index in range(1, len(this_attr) + 1):
            if index in partition and this_attr[index-1] == this_category:
                mini_partition_this_category.append(index)
                this_target_attr.append(target_attr[index-1])

        # Get unique categories of target attributes
        target_category_counts = Counter(this_target_attr).values()
        this_entropy = 0
        for count in target_category_counts:
            this_entropy += (count / len(this_target_attr)) * \
                math.log2(len(this_target_attr) / count)

        total_entropy += (this_category_counts[this_category_index] / len(
            partition)) * this_entropy

        mini_partition_this_category.clear()
        this_target_attr.clear()
        this_category_index += 1

    return total_entropy


# Outputs the output partition into a file
def output_partition_file(output_file, output_partitions):
    with open(output_file, 'w') as output:
        for partition in output_partitions:
            output.write(partition + ' ' +
                         ' '.join(map(str, output_partitions[partition])))
            if partition != list(output_partitions.keys())[-1]:
                output.write('\n')


# Driver of the program
if __name__ == "__main__":

    # Invalid number of arguments
    if(len(sys.argv) != 4):
        print(sys.argv[0], ": takes 3 arguments, not ",
              len(sys.argv)-1, ".", sep="")
        print("Expecting arguments: dataset.txt partition-input.txt partition-output.txt.")
        sys.exit(1)

    # Get input dataset and input partitions
    input_dataset, input_partitions = read_input_files(
        sys.argv[1], sys.argv[2])

    # Determine which input partition to split
    F_scores = []
    for partition in input_partitions:

        # Partition entropy
        partition_entropy = calculate_entropy(input_partitions[partition], [
                                              row[-1] for row in input_dataset])

        # Iterate through all attributes
        gain_attr = []
        for i in range(0, len(input_dataset[0])-1):
            gain_attr.append(partition_entropy - calculate_entropy_attr(input_partitions[partition], [
                             row[i] for row in input_dataset], [row[-1] for row in input_dataset]))

        # Take the max attribute (record which one it is), and determine F-score for the partition
        max_gain = max(gain_attr)
        max_attr = gain_attr.index(max_gain)

        F_scores.append(
            ((len(input_partitions[partition])/len([y for x in list(input_partitions.values()) for y in x]) * max_gain), max_attr))

    split_partition = list(input_partitions.keys())[
        [tuple[0] for tuple in F_scores].index(max([tuple[0] for tuple in F_scores]))]
    split_attr = [tuple[1] for tuple in F_scores].index(
        max([tuple[1] for tuple in F_scores]))

    # Split the partition by attribute

    # Get an array that represents an attribute column for a specific partition
    partition_split_attr = []
    for index in input_partitions[split_partition]:
        partition_split_attr.append(input_dataset[index-1][split_attr])

    # Get all unique attributes for split attribute
    unique_attrs = list(Counter(partition_split_attr).keys())

    # Partition by attribute
    new_partitions = []
    for unique_attr in unique_attrs:
        new_partition = []
        for i in range(0, len(partition_split_attr)):
            if partition_split_attr[i] == unique_attr:
                new_partition.append(input_partitions[split_partition][i])
        new_partitions.append(new_partition)

    # Put new partitions into output partition
    output_partitions = input_partitions.copy()
    output_partitions.pop(split_partition)
    i = 1
    for partition in new_partitions:
        output_partitions[split_partition + str(i)] = partition
        i += 1

    # Announce which partition was split
    print('Partition', split_partition, 'was replaced with partitions', ','.join(
        sorted(list(set(output_partitions.keys()) - set(input_partitions.keys())))), 'using Feature', split_attr+1)

    # Output to file
    output_partition_file(sys.argv[3], output_partitions)
