import pandas as pd

# PSM1 = green = right
# PSM2 = red = left
num = 28
dataset = pd.read_csv(f'ds{num}/dataset{num}.csv', header=None)
dataset.columns = ['PSM1x', 'PSM1y', 'PSM1z', 'PSM2x', 'PSM2y', 'PSM2z', 'ECMx', 'ECMy', 'ECMz', 'img_l', 'img_r']
# print(dataset.head())
dataset.reset_index(drop=True, inplace=True)
# dataset : psm1 psm2 ecm img_l img_r
# left right ends as gx gy rx ry
left = pd.read_csv(f'ds{num}/tool_img_pos_left.csv', header=None)
left.columns = ['frame_index', 'name', 'gx', 'gy', 'rx', 'ry']
right = pd.read_csv(f'ds{num}/tool_img_pos_right.csv', header=None)
print(len(dataset))
print(len(left))
print(len(right))
right.columns = ['frame_index', 'name', 'gx', 'gy', 'rx', 'ry']
psm2_x_out = pd.DataFrame({'PSMx': [], 'PSMy': [], 'PSMz': [], 'ECMx': [], 'ECMy': [], 'ECMz': []})
psm2_y_out = pd.DataFrame({'l_rx': [], 'l_ry': [], 'r_rx': [], 'r_ry': []})
psm1_x_out = pd.DataFrame({'PSMx': [], 'PSMy': [], 'PSMz': [], 'ECMx': [], 'ECMy': [], 'ECMz': []})
psm1_y_out = pd.DataFrame({'l_gx': [], 'l_gy': [], 'r_gx': [], 'r_gy': []})

left = left.sort_values(by=['frame_index'], kind='mergesort')
right = right.sort_values(by=['frame_index'], kind='mergesort')
# left.drop
psm1_rc_leftCam = left[['gx', 'gy']]
psm1_rc_rightCam = right[['gx', 'gy']]
psm2_rc_leftCam = left[['rx', 'ry']]
psm2_rc_rightCam = right[['rx', 'ry']]
psm2_x_out[['PSMx', 'PSMy', 'PSMz', 'ECMx', 'ECMy', 'ECMz']] = dataset[
    ['PSM2x', 'PSM2y', 'PSM2z', 'ECMx', 'ECMy', 'ECMz']]
psm1_x_out[['PSMx', 'PSMy', 'PSMz', 'ECMx', 'ECMy', 'ECMz']] = dataset[
    ['PSM1x', 'PSM1y', 'PSM1z', 'ECMx', 'ECMy', 'ECMz']]
left.reset_index(drop=True, inplace=True)
right.reset_index(drop=True, inplace=True)
psm2_y_out.reset_index(drop=True, inplace=True)
psm1_y_out.reset_index(drop=True, inplace=True)
# print(left.head())
# print(right.head())

psm2_y_out['l_rx'] = left['rx']
psm2_y_out['l_ry'] = left['ry']
psm2_y_out['r_rx'] = right['rx']
psm2_y_out['r_ry'] = right['ry']

psm1_y_out['l_gx'] = left['gx']
psm1_y_out['l_gy'] = left['gy']
psm1_y_out['r_gx'] = right['gx']
psm1_y_out['r_gy'] = right['gy']
psm1_y_out = psm1_y_out.replace(-1, float('nan'))
psm2_y_out = psm2_y_out.replace(-1, float('nan'))
# psm1_y_out = psm1_y_out.replace(-1, 0)
# psm2_y_out = psm2_y_out.replace(-1, 0)
# print(psm1_y_out)
# print(psm2_y_out)
# psm2_y_out = psm2_y_out.interpolate()
# psm1_y_out = psm1_y_out.interpolate()

psm1_joined = psm1_x_out.join(psm1_y_out, how='inner')
psm2_joined = psm2_x_out.join(psm2_y_out, how='inner')
psm1_joined = psm1_joined.dropna()
psm2_joined = psm2_joined.dropna()
# print(psm1_joined.head())
print(psm2_joined.shape)
print(psm2_joined.head())
psm1_x_out = psm1_joined.iloc[:, 0:6]
psm1_y_out = psm1_joined.iloc[:, 6:]
psm2_x_out = psm2_joined.iloc[:, 0:6]
psm2_y_out = psm2_joined.iloc[:, 6:]

# psm1_y_out
# psm2_x_out
# psm2_y_out
# print(psm1_x_out)
# print(psm2_x_out)
# print(psm1_y_out)
# print(psm2_y_out)

psm1_x_out.to_csv(f'./ds{num}/{num}_psm1_x.csv')
psm1_y_out.to_csv(f'./ds{num}/{num}_psm1_y.csv')
psm2_x_out.to_csv(f'./ds{num}/{num}_psm2_x.csv')
psm2_y_out.to_csv(f'./ds{num}/{num}_psm2_y.csv')