import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--remote', action='store_true', help='the code run on a server')
parser.add_argument('--num-gpu', type=int, default=1, help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=1, help='train epochs')        #默认200
parser.add_argument('--batch_size', type=int, default=16, help='batch size')

parser.add_argument('--filename', type=str, default='pems04')
parser.add_argument("--data_name", type=str, default=4, help="the number of data documents [8/4]", required=False)

parser.add_argument('--train-ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--valid-ratio', type=float, default=0.2, help='the ratio of validating dataset')
parser.add_argument('--his-length', type=int, default=24, help='the length of history time series of input')       #默认为12
parser.add_argument('--pred-length', type=int, default=12, help='the length of target time series for prediction')  #默认为12
# parser.add_argument('--sigma1', type=float, default=0.4, help='sigma for the semantic matrix')
# parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
# parser.add_argument('--thres1', type=float, default=0.9, help='the threshold for the semantic matrix')
# parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

parser.add_argument('--log', default=True, action='store_true', help='if write log to files')

parser.add_argument('--num_point', type=int, default=307, help='road Point Number [170/307] ', required=False)

args = parser.parse_args()
