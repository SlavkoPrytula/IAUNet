import numpy as np
import torch
import sys
sys.path.append("./")
from visualizations.coco_vis import visualize_masks


def create_demo_data():
    """Create sample image, masks, and bboxes for demonstration."""
    # Create a simple 512x512 grayscale image
    img = np.random.rand(512, 512) * 255
    img = img.astype(np.uint8)
    
    # Create 3 sample masks
    masks = []
    bboxes = []
    
    # Mask 1: Circle in top-left
    mask1 = np.zeros((512, 512))
    y, x = np.ogrid[:512, :512]
    center1 = (128, 128)
    radius1 = 50
    mask1[(x - center1[0])**2 + (y - center1[1])**2 <= radius1**2] = 1
    masks.append(mask1)
    
    # Corresponding bbox for mask1 (in cxcywh format, normalized)
    bbox1 = [(center1[0]) / 512, (center1[1]) / 512, 
             (2 * radius1) / 512, (2 * radius1) / 512]
    bboxes.append(bbox1)
    
    # Mask 2: Rectangle in top-right
    mask2 = np.zeros((512, 512))
    mask2[80:180, 350:450] = 1
    masks.append(mask2)
    
    # Corresponding bbox for mask2
    center2_x, center2_y = 400, 130
    width2, height2 = 100, 100
    bbox2 = [center2_x / 512, center2_y / 512, width2 / 512, height2 / 512]
    bboxes.append(bbox2)
    
    # Mask 3: Ellipse in bottom center
    mask3 = np.zeros((512, 512))
    center3 = (256, 380)
    a, b = 80, 40  # Semi-major and semi-minor axes
    mask3[((x - center3[0])**2 / a**2 + (y - center3[1])**2 / b**2) <= 1] = 1
    masks.append(mask3)
    
    # Corresponding bbox for mask3
    bbox3 = [center3[0] / 512, center3[1] / 512, (2 * a) / 512, (2 * b) / 512]
    bboxes.append(bbox3)

    # Mask 4: Overlapping circle
    mask4 = np.zeros((512, 512))
    y, x = np.ogrid[:512, :512]
    center4 = (350, 150)
    radius4 = 50
    mask4[(x - center4[0])**2 + (y - center4[1])**2 <= radius4**2] = 1
    masks.append(mask4)
    
    # Corresponding bbox for mask1 (in cxcywh format, normalized)
    bbox4 = [(center4[0]) / 512, (center4[1]) / 512, 
             (2 * radius4) / 512, (2 * radius4) / 512]
    bboxes.append(bbox4)
    
    # Convert to tensors
    masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32)
    bboxes_tensor = torch.tensor(np.array(bboxes), dtype=torch.float32)
    
    return img, masks_tensor, bboxes_tensor


def demo_visualization():
    """Demonstrate the enhanced visualization features."""
    print("Creating demo data...")
    img, masks, bboxes = create_demo_data()
    
    target_shape = (512, 512)
    
    print("Generating visualizations...")
    
    # # Demo 1: Basic mask visualization
    # print("1. Basic mask visualization (no borders, no bboxes)")
    # visualize_masks(
    #     img=img,
    #     masks=masks,
    #     shape=target_shape,
    #     alpha=0.6,
    #     path="./demo_basic_masks.png",
    #     show_img=True,
    #     figsize=[15, 7]
    # )
    
    # # Demo 2: Mask visualization with borders
    # print("2. Mask visualization with colored borders")
    # visualize_masks(
    #     img=img,
    #     masks=masks,
    #     shape=target_shape,
    #     alpha=0.6,
    #     draw_border=True,
    #     border_size=3,
    #     path="./demo_masks_with_borders.png",
    #     show_img=True,
    #     figsize=[15, 7]
    # )
    
    # # Demo 3: Mask visualization with matching bboxes
    # print("3. Mask visualization with matching colored bboxes")
    # visualize_masks(
    #     img=img,
    #     masks=masks,
    #     shape=target_shape,
    #     alpha=0.6,
    #     bboxes=bboxes,
    #     bbox_linewidth=3,
    #     path="./demo_masks_with_bboxes.png",
    #     show_img=True,
    #     figsize=[15, 7]
    # )
    
    # # Demo 4: Full visualization with borders and bboxes
    print("4. Full visualization with borders and matching bboxes")
    visualize_masks(
        img=img,
        masks=masks,
        shape=target_shape,
        alpha=0.6,
        draw_border=True,
        border_size=15,
        bboxes=bboxes,
        bbox_linewidth=15,
        path="./demo_full_visualization.png",
        show_img=False,
        figsize=[30, 30]
    )
    
    # Demo 5: Static colors for reproducible results
    # print("5. Static colors for reproducible visualization")
    # visualize_masks(
    #     img=img,
    #     masks=masks,
    #     shape=target_shape,
    #     alpha=0.6,
    #     static_color=True,
    #     draw_border=True,
    #     border_size=20,
    #     bboxes=bboxes,
    #     bbox_linewidth=20,
    #     path="./demo_static_colors.png",
    #     show_img=False,
    #     figsize=[30, 30]
    # )
    
    print("Demo complete! Check the generated PNG files:")
    print("- demo_basic_masks.png")
    print("- demo_masks_with_borders.png") 
    print("- demo_masks_with_bboxes.png")
    print("- demo_full_visualization.png")
    print("- demo_static_colors.png")


if __name__ == "__main__":
    demo_visualization()
