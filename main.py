from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# โหลด pipeline (ใช้โมเดลที่ fine-tuned เป็น Ghibli style จาก HuggingFace)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "nitrosocke/ghibli-diffusion",  # หรือโมเดลอื่นที่คุณโหลดมา local ก็ได้
    torch_dtype=torch.float32 # ใช้ float32 แทน float16 เพื่อความเข้ากันได้กับ CPU,
).to("cpu")

# โหลดภาพต้นฉบับ (หรือใช้ Image.open("your-image.jpg"))
init_image = Image.open("your_input.jpg").convert("RGB")
init_image = init_image.resize((768, 512))  # ปรับขนาดตามโมเดล

# Prompt ที่ใช้
prompt = "Convert this image to Studio Ghibli style illustration a peaceful anime village in Studio Ghibli style, soft colors, magical lighting, dreamy landscape "

# แปลงภาพ
# strength: ควบคุม “ความเปลี่ยนแปลง” จากภาพต้นฉบับ (0.3 = เปลี่ยนน้อย, 0.8 = เปลี่ยนเยอะ)
# guidance_scale: ควบคุมความตรงกับ prompt (7.5 = สูง, 1.0 = ต่ำ)
image = pipe(prompt=prompt, image=init_image, strength=0.3, guidance_scale=7.5).images[0]

# บันทึกผลลัพธ์
image.save("ghibli_output.png")
print("✅ แปลงภาพเสร็จแล้ว: ghibli_mosalah_output.png")
