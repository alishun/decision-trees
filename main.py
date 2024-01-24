import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

AXIS_NUM = 7
RESULT_NUM = 4
BOX_WIDTH = 1600
BOX_HEIGHT = 200
AXIS_BOND = 4000
MAX_DEPTH = 999
CROSS_FOLDS = 10


# When a condition is greater or equal to, it goes to the right
# When a condition is less than, it goes to the left

class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.depth = 0

    def apply(self, in_data):
        pass

    def is_leaf(self):
        return self.left is None and self.right is None


class TreeNodeCondition(TreeNode):
    def __init__(self, cond_axis, cond_val, left, right, depth):
        super().__init__()
        self.cond_axis = cond_axis
        self.cond_val = cond_val
        self.left = left
        self.right = right
        self.depth = depth

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def apply(self, in_data):
        if in_data[self.cond_axis] >= self.cond_val:
            return self.right.apply(in_data)
        else:
            return self.left.apply(in_data)

    def __str__(self):
        rep = str(self.cond_val)
        if self.is_leaf():
            return rep
        rep += "{ " + str(self.left) + " }" if self.left is not None else "{}"
        rep += "{ " + str(self.right) + " }" if self.left is not None else "{}"
        return rep


class TreeNodeResult(TreeNode):
    def __init__(self, result, depth):
        super().__init__()
        self.result = int(result)
        self.depth = depth

    def apply(self, in_data):
        return self.result

    def __str__(self):
        return str(self.result)


def pre_processing(data, fold):
    np.random.shuffle(data)
    divided_arrays = np.split(data, fold)
    return divided_arrays


def all_same(train_set):
    one_value = train_set[0][-1]
    for i in range(len(train_set)):
        if train_set[i][-1] != one_value:
            return False
    return True


def visualize_tree(decision_tree):
    figure = plt.figure()
    figure.set_figwidth(100)
    figure.set_figheight(150)
    axis = figure.add_subplot(1, 1, 1)
    top_x = 0
    top_y = AXIS_BOND - 100

    visualize_recurse(decision_tree, axis, top_x, top_x, -AXIS_BOND, AXIS_BOND)
    plt.xlim([-AXIS_BOND, AXIS_BOND])
    plt.ylim([-AXIS_BOND, AXIS_BOND])
    axis.set_axis_off()
    plt.show()


def visualize_recurse(tree, axis, org_x, org_y, left_bound, right_bound):
    if tree is None:
        return
    tag = "[ X" + str(tree.cond_axis) + " < " + str(tree.cond_val) + " ]" \
        if not (tree.is_leaf()) \
        else "leaf:" + str(tree.result)
    visualize_box(axis, org_x, org_y, BOX_WIDTH, BOX_HEIGHT, tag)
    next_y = org_y - 400
    left_x = 0.5 * (org_x + left_bound)
    right_x = 0.5 * (org_x + right_bound)
    if tree.left is not None:
        visualize_line(org_x + BOX_WIDTH / 2, org_y, left_x + BOX_WIDTH / 2, next_y + BOX_HEIGHT)
    if tree.right is not None:
        visualize_line(org_x + BOX_WIDTH / 2, org_y, right_x + BOX_WIDTH / 2, next_y + BOX_HEIGHT)
    visualize_recurse(tree.left, axis, left_x, next_y, left_bound, org_x)
    visualize_recurse(tree.right, axis, right_x, next_y, org_x, right_bound)

def visualize_line(x1, y1, x2, y2):
    plt.plot([x1, x2], [y1, y2])


def visualize_box(axis, x, y, width, height, text):
    #rect = patches.Rectangle((x, y), width * 3, height * 2, linewidth=0.3, fill=False)
    axis.annotate(text, (0.5 * (2 * x + width), 0.5 * (2 * y + height)), ha="center", va="center", fontsize=10)
    #axis.add_patch(rect)


def get_room(data, room):
    return data[data[:, 7] == room]


def visualize_room(graph, data, room):
    graph.boxplot(get_room(data, room)[:, :7], patch_artist=True, showfliers=True)
    graph.set(title=f'Room {room}', xlabel="Router No.", ylabel="Signal DBm")


def visualize_data(data, name):
    graph, (room1, room2, room3, room4) = plt.subplots(1, 4)
    graph.set_figheight(8)
    graph.set_figwidth(36)
    graph.suptitle(f'{name} data', fontsize=8)

    visualize_room(room1, data, 1)
    visualize_room(room2, data, 2)
    visualize_room(room3, data, 3)
    visualize_room(room4, data, 4)
    plt.show()


def get_h(data):
    unique_classes, class_counts = np.unique(data, return_counts=True)
    probabilities = class_counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))


def gain(total, left, right):
    remainder = len(left) / (len(left) + len(right)) * get_h(left) + \
                len(right) / (len(left) + len(right)) * get_h(right)
    return get_h(total) - remainder


def find_split(train_set):
    max_entropy = 0
    max_split = (None, None, None, None)
    set_len = len(train_set)

    for feature in range(AXIS_NUM):
        for i in range(set_len):
            # print(f"\r{feature * set_len + i}/{AXIS_NUM * set_len}", end="")
            left_set, right_set = split_set(train_set, feature, train_set[i][feature])
            entropy = gain(train_set, left_set, right_set)
            if entropy >= max_entropy:
                max_entropy = entropy
                max_split = (feature, train_set[i][feature], left_set, right_set)
    # print("\r", end="")
    return max_split


def split_set(train_set, axis, val):
    left_set = []
    right_set = []
    for row in train_set:
        if row[axis] >= val:
            right_set.append(row)
        else:
            left_set.append(row)
    return np.array(left_set), np.array(right_set)


def most_nodes(train_set):
    return np.argmax(np.bincount(train_set[:, -1].astype(int)))


def decision_tree_learning(train_set, depth_left):
    print(f'\rTraining on the {MAX_DEPTH - depth_left} depth', end="")
    if len(train_set) == 0:
        return TreeNodeResult(0, MAX_DEPTH - depth_left)
    elif all_same(train_set):
        # print('All same', train_set[0][-1])
        return TreeNodeResult(train_set[0][-1], MAX_DEPTH - depth_left)
    elif depth_left == 0:
        mode = most_nodes(train_set)
        # print('Max depth reached, return', mode)
        return TreeNodeResult(mode, MAX_DEPTH - depth_left)
    else:
        (axis, val, left_set, right_set) = find_split(train_set)
        # print('Splitting on axis', axis, 'at', val)
        return TreeNodeCondition(axis, val, decision_tree_learning(left_set, depth_left - 1),
                                 decision_tree_learning(right_set, depth_left - 1), MAX_DEPTH - depth_left)


def evaluate_tree(decision_tree, test_set):
    evaluation_per_class = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_tn = 0
    for i in range(1, RESULT_NUM + 1):
        evaluation_per_class[i] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    for row in test_set:
        expected = int(row[-1])
        actual = int(decision_tree.apply(row))
        if expected == actual:
            evaluation_per_class[expected]['tp'] += 1
            overall_tp += 1
            for i in range(1, RESULT_NUM + 1):
                if i != expected:
                    evaluation_per_class[i]['tn'] += 1
                    overall_tn += 1
        else:
            evaluation_per_class[expected]['fn'] += 1
            overall_fn += 1
            evaluation_per_class[actual]['fp'] += 1
            overall_fp += 1
    metrics_per_class = {}
    for i in range(1, RESULT_NUM + 1):
        metrics_per_class[i] = {}
        if evaluation_per_class[i]['tp'] == 0:
            metrics_per_class[i]['precision'] = 0
            metrics_per_class[i]['recall'] = 0
            metrics_per_class[i]['f1'] = 0
        else:
            metrics_per_class[i]['precision'] = (evaluation_per_class[i]['tp']
                                                 / (evaluation_per_class[i]['tp'] + evaluation_per_class[i]['fp']))
            metrics_per_class[i]['recall'] = (evaluation_per_class[i]['tp']
                                              / (evaluation_per_class[i]['tp'] + evaluation_per_class[i]['fn']))
            metrics_per_class[i]['f1'] = (2 * metrics_per_class[i]['precision']
                                          * metrics_per_class[i]['recall'] / (metrics_per_class[i]['precision']
                                        + metrics_per_class[i]['recall']))

        # print(evaluation_per_class[i])
    evaluation_overall = {'tp': overall_tp / RESULT_NUM,
                          'fp': overall_fp / RESULT_NUM,
                          'fn': overall_fn / RESULT_NUM,
                          'tn': overall_tn / RESULT_NUM,
                          'accuracy': overall_tp / len(test_set)}

    return metrics_per_class, evaluation_overall


def cross_validation_train(data, folds, evaluation_per_class = {}, evaluation_overall = {}):
    shuffled_datasets = pre_processing(data, folds)
    for i in range(CROSS_FOLDS):
        print(f"\nTraining on fold {i + 1}/{CROSS_FOLDS}")
        test_set = shuffled_datasets[i]
        train_set = np.concatenate(shuffled_datasets[:i] + shuffled_datasets[i + 1:])
        decision_tree = decision_tree_learning(train_set, MAX_DEPTH)
        i_evaluation_per_class, i_evaluation_overall = evaluate_tree(decision_tree, test_set)
        if not evaluation_per_class:
            evaluation_per_class = i_evaluation_per_class
        else:
            for j in range(1, RESULT_NUM + 1):
                evaluation_per_class[j]['precision'] += i_evaluation_per_class[j]['precision']
                evaluation_per_class[j]['recall'] += i_evaluation_per_class[j]['recall']
                evaluation_per_class[j]['f1'] += i_evaluation_per_class[j]['f1']
        if not evaluation_overall:
            evaluation_overall = i_evaluation_overall
        else:
            evaluation_overall['accuracy'] += i_evaluation_overall['accuracy']
            evaluation_overall['tp'] += i_evaluation_overall['tp']
            evaluation_overall['fp'] += i_evaluation_overall['fp']
            evaluation_overall['tn'] += i_evaluation_overall['tn']
            evaluation_overall['fn'] += i_evaluation_overall['fn']
    return evaluation_per_class, evaluation_overall

def main():
    clean_data_path = 'wifi_db/clean_dataset.txt'
    noisy_data_path = 'wifi_db/noisy_dataset.txt'
    clean_data = np.loadtxt(clean_data_path)
    noisy_data = np.loadtxt(noisy_data_path)
    shuffled_datasets = pre_processing(clean_data, CROSS_FOLDS)
    visualize_data(clean_data, "Clean")
    visualize_data(noisy_data, "Noisy")

    clean_decision_tree = decision_tree_learning(clean_data, MAX_DEPTH)
    visualize_tree(clean_decision_tree)

    evaluation_per_class = {}
    evaluation_overall = {}

    evaluation_per_class, evaluation_overall = (
        cross_validation_train(clean_data, CROSS_FOLDS, evaluation_per_class, evaluation_overall))
    evaluation_per_class, evaluation_overall = (
        cross_validation_train(noisy_data, CROSS_FOLDS, evaluation_per_class, evaluation_overall))

    for i in range(1, RESULT_NUM + 1):
        evaluation_per_class[i]['precision'] = evaluation_per_class[i]['precision'] / (2 * CROSS_FOLDS)
        evaluation_per_class[i]['recall'] = evaluation_per_class[i]['recall'] / (2 * CROSS_FOLDS)
        evaluation_per_class[i]['f1'] = evaluation_per_class[i]['f1'] / (2 * CROSS_FOLDS)
        print(i, evaluation_per_class[i])

    evaluation_overall['accuracy'] = evaluation_overall['accuracy'] / (2 * CROSS_FOLDS)
    evaluation_overall['tp'] = evaluation_overall['tp'] / (2 * CROSS_FOLDS)
    evaluation_overall['fp'] = evaluation_overall['fp'] / (2 * CROSS_FOLDS)
    evaluation_overall['tn'] = evaluation_overall['tn'] / (2 * CROSS_FOLDS)
    evaluation_overall['fn'] = evaluation_overall['fn'] / (2 * CROSS_FOLDS)

    print(evaluation_overall)

if __name__ == "__main__":
    main()