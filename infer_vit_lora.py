# infer_vit_lora.py
import argparse, json, torch, timm
from PIL import Image
from torchvision import transforms
from train_vit_lora import apply_lora_to_vit, freeze_all_but_norm_and_head

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best_vit_lora_*.pt")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device)
    tri_args  = state['args']; classes = state['class_names']

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
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = logits.softmax(-1).cpu().numpy()[0]
    idx = int(prob.argmax())
    out = {"pred": classes[idx], "confidence": float(prob[idx]),
           "probabilities": {c: float(p) for c,p in zip(classes, prob)}}
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
