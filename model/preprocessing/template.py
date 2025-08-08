17*39 window for each digit
cv2.TM_CCOEFF_NORMED

3 digits positions:
x1-x2: 40-60, 60-79, 79-96
y1-y2: 7-49

template matching in three regions, compare with 10 digits in model/preprocessing/templates folder,
and check which one digit matches with max similarity with TM_CCOEFF_NORMED.

