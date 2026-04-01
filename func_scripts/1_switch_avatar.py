import argparse
import os
def main(args):
    avatar_dir = os.path.abspath(os.path.join("./linked_data/avatars", args.name))
    if not os.path.exists(avatar_dir):
        print(f"Avatar {args.name} does not exist.")
        return
    # DDSP-SVC
    DDSP_dir = os.path.abspath("./third_party/DDSP-SVC")
    ##remove old link
    if os.path.islink(os.path.join(DDSP_dir, "data")):
        os.unlink(os.path.join(DDSP_dir, "data"))
    if os.path.islink(os.path.join(DDSP_dir, "exp")):
        os.unlink(os.path.join(DDSP_dir, "exp"))
    ##relink
    os.symlink(os.path.join(avatar_dir, "model_data", "DDSP-SVC", "data"), 
                os.path.join(DDSP_dir, "data"))
    os.symlink(os.path.join(avatar_dir, "model_data", "DDSP-SVC", "ckpt"), 
                os.path.join(DDSP_dir, "exp"))

    
    # SO-VITS-SVC
    sovits_dir = os.path.abspath("./third_party/SO-VITS-SVC")
    ##remove old link
    if os.path.islink(os.path.join(sovits_dir, "dataset", "44k")):
        os.unlink(os.path.join(sovits_dir, "dataset", "44k"))
    if os.path.islink(os.path.join(sovits_dir, "logs", "44k")):
        os.unlink(os.path.join(sovits_dir, "logs", "44k"))
    ##relink
    os.symlink(os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "data"), 
                os.path.join(sovits_dir, "dataset", "44k"))
    os.symlink(os.path.join(avatar_dir, "model_data", "SO-VITS-SVC", "ckpt"), 
                os.path.join(sovits_dir, "logs", "44k"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="name of the avatar")
    args = parser.parse_args()
    main(args)