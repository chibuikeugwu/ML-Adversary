import argparse

def prediction_args():

    parser = argparse.ArgumentParser(description='predictions')
    ###positional
    
    parser.add_argument("path_to_image",default="/home/cugwu_dg/cpts-528-project/528Project/dog_example2.jpg", help = "choose your path" )
    parser.add_argument("model",default="/home/cugwu_dg/cpts-528-project/528Project/model.pth", help = "choose your stored model")

    ###optional
    parser.add_argument("--top_k", default = 3, type = int, help = "top k classes")
    parser.add_argument("--gpu", default = True, action = "store_true", help = "gpu is False by default")






    ###
    parser.parse_args()
    return parser




def main():
    print("this is prediction arguments")

if __name__== "__main__":
    main()

