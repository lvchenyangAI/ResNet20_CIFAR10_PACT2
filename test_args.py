import argparse
parser = argparse.ArgumentParser(description="命令行传入一个数据")
parser.add_argument('integers', type=str, help='传入的数字')
args = parser.parse_args()
print(args)