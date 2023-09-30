import argparse



def train_args():

    parser = argparse.ArgumentParser(description = "train and save your model")

    ###positional
    
    ###optional
    parser.add_argument("--save_dir", action="store", default=".", type = str, help = "directory for the trained model")
    parser.add_argument("--num_classes", default = 120, type = int, help = "number of classes")

    parser.add_argument("--arch", default = "densenet121", help = "architecture: densenet121 (by default) or vgg11")
    parser.add_argument("--learning_rate", default = 0.001, help = "learning rate")
    parser.add_argument("--hidden_units", default = 500, help = "number of hidden units for the last layer")
    parser.add_argument("--epochs", default = 10, type = int, help = "number of epochs")
    parser.add_argument("--gpu", default = True, action = "store_true", help = "gpu is False by default")


    ###
    parser.parse_args()
    return parser



def main():
    print("this is train arguments")

if __name__== "__main__":
    main()
