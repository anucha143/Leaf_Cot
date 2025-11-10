# pipeline_siglip_vit.py
import argparse, json, torch, timm
from PIL import Image
from torchvision import transforms
from siglip_leaf_gate import SiglipLeafGate
from train_vit_lora import apply_lora_to_vit, freeze_all_but_norm_and_head

def load_tri_model(ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    tri_args  = state['args']
    classes = state['class_names']

    try:
        model = timm.create_model(tri_args['model'], pretrained=True, num_classes=len(classes))
    except Exception:
        model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=len(classes))

    apply_lora_to_vit(model, r=tri_args['lora_r'], alpha=tri_args['lora_alpha'], lora_dropout=tri_args['lora_dropout'])
    freeze_all_but_norm_and_head(model)
    model.load_state_dict(state['model_state'])
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize(int(tri_args['img_size']*1.15)),
        transforms.CenterCrop(tri_args['img_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    return model, classes, tf, tri_args

@torch.no_grad()
def tri_classify(model, tf, classes, image_path: str, device: str):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = logits.softmax(-1).cpu().numpy()[0]
    k = int(prob.argmax())
    return {"tri_class": classes[k], "tri_confidence": float(prob[k]),
            "tri_probabilities": {c: float(p) for c,p in zip(classes, prob)}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--tri_ckpt", required=True, help="Path to best_vit_lora_*.pt")
    ap.add_argument("--siglip_name", default="google/siglip-so400m-patch14-384")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate = SiglipLeafGate(model_name=args.siglip_name, device=device)
    g = gate.predict(args.image, threshold=args.threshold)

    if not g["pass_gate"]:
        print(json.dumps({
            "is_leaf": False,
            "leaf_confidence": g["p_leaf"],
            "threshold": args.threshold,
            "decision": "Not a leaf / rejected at gate",
            "image": g["image"]
        }, ensure_ascii=False, indent=2))
        return

    model, classes, tf, tri_args = load_tri_model(args.tri_ckpt, device)
    t = tri_classify(model, tf, classes, args.image, device)
    out = {
        "is_leaf": True,
        "leaf_confidence": g["p_leaf"],
        "threshold": args.threshold,
        "decision": "Accepted at gate",
        **t
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
