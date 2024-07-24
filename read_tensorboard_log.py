import tensorflow as tf
import sys

def read_tensorboard_log(file_path):
    try:
        for summary in tf.compat.v1.train.summary_iterator(file_path):
            print(f"Step: {summary.step}")
            for value in summary.summary.value:
                print(f"  Tag: {value.tag}, Value: {value.simple_value}")
            break  # Just print the first event for brevity
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    read_tensorboard_log(file_path)