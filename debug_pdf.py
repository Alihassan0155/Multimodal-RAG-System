import fitz, sys
from pathlib import Path

pdf_path = r"d:\Projects\Multimodal-rag\data\uploads\Attention-is-all-you-need.pdf"
doc = fitz.open(pdf_path)

for page_num in range(min(10, len(doc))):
    page = doc[page_num]
    text = page.get_text("text").strip()
    drawings = page.get_drawings()
    images = page.get_images(full=True)
    xobjects = page.get_xobjects()
    
    # Classify drawings by type
    thin_h = sum(1 for d in drawings if d.get("rect") and d["rect"].height < 3 and d["rect"].width > 10)
    thin_v = sum(1 for d in drawings if d.get("rect") and d["rect"].width < 3 and d["rect"].height > 10)
    shapes_gt5 = sum(1 for d in drawings if d.get("rect") and d["rect"].width > 5 and d["rect"].height > 5)
    shapes_gt30 = sum(1 for d in drawings if d.get("rect") and d["rect"].width > 30 and d["rect"].height > 30)
    
    # Area calculation  
    page_area = page.rect.width * page.rect.height
    draw_area_gt30 = sum(d["rect"].width * d["rect"].height for d in drawings if d.get("rect") and d["rect"].width > 30 and d["rect"].height > 30)
    draw_area_gt5 = sum(d["rect"].width * d["rect"].height for d in drawings if d.get("rect") and d["rect"].width > 5 and d["rect"].height > 5)
    
    # Sample some drawing rects on figure-heavy pages
    sample_rects = [(round(d["rect"].width,1), round(d["rect"].height,1), d.get("fill"), d.get("color")) 
                    for d in drawings[:5] if d.get("rect")]
    
    import re
    caption = re.findall(r"(?m)^(?:Figure|Fig\.?)\s*\d+[:\.]?\s*(.{5,150})", text)
    caption_with_space = re.findall(r"(?m)^\s*(?:Figure|Fig\.?)\s*\d+[:\.]?\s*(.{5,150})", text)
    
    print(f"\n=== PAGE {page_num+1} ===")
    print(f"  text_chars: {len(text)}")
    print(f"  raster_images: {len(images)}")
    print(f"  xobjects: {len(xobjects)}")
    print(f"  drawings total: {len(drawings)}")
    print(f"  thin_h_lines: {thin_h}, thin_v_lines: {thin_v}")
    print(f"  shapes_gt5: {shapes_gt5}, shapes_gt30: {shapes_gt30}")
    print(f"  draw_area_gt30/page_area: {draw_area_gt30/page_area:.4f}")
    print(f"  draw_area_gt5/page_area: {draw_area_gt5/page_area:.4f}")
    print(f"  caption (strict ^): {caption}")
    print(f"  caption (with spaces): {caption_with_space}")
    print(f"  sample drawing rects (w,h,fill,color): {sample_rects}")

doc.close()
