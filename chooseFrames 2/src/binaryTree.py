import matplotlib.pyplot as plt

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def insert(root, data):
    if root is None:
        return Node(data)
    else:
        if data < root.data:
            root.left = insert(root.left, data)
        else:
            root.right = insert(root.right, data)
    return root

def plot_tree(node, x, y, dx, dy):
    if node is not None:
        plt.text(x, y, str(node.data), fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))
        if node.left:
            plt.plot([x, x - dx], [y - dy, y - 1], 'k-')
            plot_tree(node.left, x - dx, y - 1, dx / 2, dy)
        if node.right:
            plt.plot([x, x + dx], [y - dy, y - 1], 'k-')
            plot_tree(node.right, x + dx, y - 1, dx / 2, dy)

# Example usage:

def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print('   ' * level + '->', node.data)
        print_tree(node.left, level + 1)
def main():
    numbers = [5, 3, 8, 2, 4, 7, 9]
    root = None
    for number in numbers:
        root = insert(root, number)

    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plot_tree(root, 0, 0, 6, 1)
    plt.show()
    print_tree(root)


if __name__ == "__main__":
    main()
