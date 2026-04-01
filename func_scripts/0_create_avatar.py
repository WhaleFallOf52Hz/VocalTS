
# !!!!!!!!!!!!!!!!!Please use this script in the base directory of the project!!!!!!!!!!!!!!!
import argparse
import os
def main(args):
    
    avatar_dir = os.path.abspath(os.path.join("./linked_data/avatars", args.name))
    if os.path.exists(avatar_dir):
        print(f"Avatar {args.name} already exists. Please choose a different name.")
        return
    os.makedirs(avatar_dir)
    #meta data
    os.makedirs(os.path.join(avatar_dir, "meta_data", "audio"))
    os.makedirs(os.path.join(avatar_dir, "meta_data", "vocal_sliced"))
    os.makedirs(os.path.join(avatar_dir, "meta_data", "MSST"))
    os.makedirs(os.path.join(avatar_dir, "meta_data", "ncm"))
    

    #DDSP-SVC
    os.makedirs(os.path.join(avatar_dir, "model_data", "DDSP-SVC", "data"))
    os.makedirs(os.path.join(avatar_dir, "model_data", "DDSP-SVC", "ckpt"))

    os.symlink(os.path.join(avatar_dir, "meta_data", "vocal_sliced"), 
                os.path.join(avatar_dir, "model_data", "DDSP-SVC", "data", "audio"))

    #SO-VITS-SVC
    os.makedirs(os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "data"))
    os.makedirs(os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "ckpt", "diffusion"))

    os.symlink(os.path.join(avatar_dir, "meta_data", "vocal_sliced"), 
                os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "data", "audio"))
    
    
    sovits_dir = os.path.abspath("./third_party/SO-VITS-SVC")

    os.symlink(os.path.join(sovits_dir, "universial_model","model_0.pt"), 
                os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "ckpt", "diffusion","model_0.pt"))
    os.symlink(os.path.join(sovits_dir, "universial_model","G_0.pth"), 
                os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "ckpt", "G_0.pth"))
    os.symlink(os.path.join(sovits_dir, "universial_model","D_0.pth"), 
                os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "ckpt", "D_0.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="name of the avatar")
    args = parser.parse_args()
    main(args)
    
