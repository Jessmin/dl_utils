def calculate_IOU(rec1, rec2):
    """ 计算两个矩形框的交并比
    Args:
    	rec1: [left1,top1,right1,bottom1]  # 其中(left1,top1)为矩形框rect1左上角的坐标，(right1, bottom1)为右下角的坐标，下同。
     	rec2: [left2,top2,right2,bottom2]
     	
    Returns: 
    	交并比IoU值
    """
    left_max = max(rec1[0], rec2[0])
    top_max = max(rec1[1], rec2[1])
    right_min = min(rec1[2], rec2[2])
    bottom_min = min(rec1[3], rec2[3])
    # 两矩形相交时计算IoU
    if (left_max < right_min
            or bottom_min > top_max):  # 判断时加不加=都行，当两者相等时，重叠部分的面积也等于0
        rect1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        rect2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        area_cross = (bottom_min - top_max) * (right_min - left_max)
        return area_cross / rect2_area
    else:
        return 0
