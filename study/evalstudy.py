import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Read the CSV file
file_path = '/home/dai/GPU-Student-2/Cederic/DataSciPro/study/studyentries.csv'
df = pd.read_csv(file_path)
print('Done reading file')

# ------------------ Evaluate the Image Selection Task ------------------
def mean_for_outliers(columns, user_selection_data):
    mean = 0
    num_images = len(columns)
    for column in columns:
        mean += user_selection_data[column]
    mean = mean / num_images
    return mean

def evaluate_Selection(df: pd.DataFrame):
    
    all_columns = df.columns
    desired_columns = [col for col in all_columns if "ImageSelection" in col and "selection" in col]
    print('\033[35m' +'Columns evaluated for the image selection task: ' + str(desired_columns) + '\033[0m')

    user_selection_lst = [] # list[df.list]; every user has a list of selected images
    image_selection_lst = [] # list[str]; data of selected images

    for user in range(df.shape[0]):
        user_selection_data = df.iloc[user][desired_columns]

        # Check for outliers
        mean = mean_for_outliers(desired_columns, user_selection_data)
        print('\033[34m' + 'Mean of user ' + str(user) + ': ' + str(round(mean, 3)) + '\033[0m')
        if mean == 0:
            print('\033[31m' + 'User ' + str(user) + ' is an outlier' + '\033[0m')
            continue
        user_selection_lst.append(user_selection_data)

    print('\033[32m' + 'Total number of valid users: ' + str(len(user_selection_lst)) + '\033[0m')

    
    for column in desired_columns:
        image_selection = []
        for user in range(len(user_selection_lst)):
            image_selection.append(user_selection_lst[user][column])
        image_selection_lst.append(image_selection)

    return user_selection_lst, image_selection_lst

def plot_histogram(image_selection_lst, which_image, output_path):

    fig, ax = plt.subplots(figsize=(10, 6))
    num_bins =  range(min(image_selection_lst[which_image]), (max(image_selection_lst[which_image]))+2)
    n, bins, patches = ax.hist(image_selection_lst[which_image], bins=num_bins, edgecolor='white', linewidth=1)
    # Farben festlegen
    for i, patch in enumerate(patches):
        plt.setp(patch, 'facecolor', plt.cm.gist_heat(i / len(patches)))
    # Balkenbeschriftungen hinzufügen
    for i in range(len(patches)):
        ax.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height() + 0.1, 
                str(int(patches[i].get_height())), ha='center', fontsize=12, color='black')
    # Beschriftungen und Titel hinzufügen
    ax.set_title('Histogram of \'human assumed\' classifier switch', fontsize=20, pad=20)
    ax.set_xlabel('chosen image', fontsize=15, labelpad=15)
    ax.set_ylabel('number of participants\nchosen this image', fontsize=15, labelpad=15)

    ax.set_xticks(bins[:-1] + 0.5)  # Setzt die Ticks in die Mitte der Balken
    ax.set_xticklabels([str(int(tick)) for tick in bins[:-1]])  # Beschriftungen anpassen

    # Achsenbegrenzungen setzen
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, max(n) + 1)
    # Grid und Hintergrund einstellen
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f0f0f0')
    # Layout anpassen und Plot anzeigen

    custom_text = f'-1: \'none is smiling\'\n0-{max(image_selection_lst[which_image])}: image number'
    plt.text(0.95, 0.95, custom_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    #plt.show()


# ------------------ Evaluate the Annotation Task ------------------
def visualize_mask(mask_array: np.ndarray, image_path: str, num_of_participants: int, output_path: str):
    import matplotlib.pyplot as plt
    import cv2

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    mask = cv2.resize(mask_array, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    fig, ax = plt.subplots()
    cmap = 'gist_heat_r'

    ax.imshow(image, interpolation='nearest')
    ax.imshow(mask, cmap=cmap, alpha=0.5, interpolation='nearest')
    
    plt.title(f'human relevance heatmap\n of all (N = {num_of_participants}) participants')
    plt.colorbar(ax.imshow(mask, cmap=cmap, alpha=0.5, interpolation='nearest'))
    plt.axis('off')  # Turn off the axis
    plt.savefig(output_path)
    plt.close(fig)
    #plt.show()

def create_mask(polygons: list, image_shape: tuple):
    import numpy as np
    import cv2

    # returns a mask with the polygons filled as 1s all others are 0s

    mask = np.zeros(image_shape, dtype=np.uint8)
    for polygon in polygons:
        polygon = np.array(polygon)
        cv2.fillPoly(mask, [polygon], 1)
        #visualize_mask(mask)
    print('\033[31m' + 'Polygon mask with all polygons is created! ' + '\033[0m')
    return mask


def evaluate_annotation(df: pd.DataFrame, which_image: int, image_path: str, output_path: str):
    import numpy as np

    all_columns = df.columns
    desired_columns = [col for col in all_columns if "ImageAnnotations" in col and str(which_image) in col]
    print('\033[35m' +'Columns evaluated for the image selection task: ' + str(desired_columns) + '\033[0m')

    # here the range needs to be adjusted based on how many polygons the user created,
    # TO DO: make this dynamic!!!!
    all_mask_colums = []
    for i in range(10):
        mask_colums = [col for col in desired_columns if f"{str(i)}/points/" in col]
        all_mask_colums.append(mask_colums)
    print('\033[34m' +'Separated Columns evaluated for the image selection task: ' + str(all_mask_colums) + '\033[0m')
    # keys for points for polygon masks are stored in a global list like list[list[str]]

    average_mask = np.zeros((256, 256), dtype=np.uint8)
    # for every user evaluate
    for user in range(0, df.shape[0]):
        # for every polygon one user created evaluate
        polygons_lst = []
        for polygon_idx in range(len(all_mask_colums)):
            polygon_coords = df.iloc[user][all_mask_colums[polygon_idx]] 
            # tuple consists of (x,y) coordinates of the polygon
            tuple_polygon_coords = [(polygon_coords[i], polygon_coords[i+1]) for i in range(0, len(polygon_coords), 2)]
            # important step: rescale the coordinates to 256x256 from the study resolution 512x512
            filtered_tuple_polygon_list = [(int(tup[0] / 2), int(tup[1] / 2)) for tup in tuple_polygon_coords if not any(np.isnan(val) for val in tup)]
            polygons_lst.append(filtered_tuple_polygon_list)
        
        # filter polygons, not really dynamic but works
        polygons_lst = [lst for lst in polygons_lst if len(lst) > 0]
        print('\033[32m' + 'User ' + str(user) + f' created (number of polygons: {len(polygons_lst)}) the following polygons: ' + str(polygons_lst) + '\033[0m')
        # create the mask
        image_shape = (256, 256)
        mask = create_mask(polygons_lst, image_shape)
        average_mask += mask
        #if user % 5 == 0:
        #    visualize_mask(average_mask, image_path, user+1)
    
    visualize_mask(average_mask, image_path, df.shape[0], output_path)


if __name__ == '__main__':
    # ------------------ Evaluate the Image Selection Task ------------------
    which_image = 0
    image_idxs = [27300, 27398, 27591, 27931, 28113, 28125, 28285, 28362, 28383, 28583, 28782, 28892, 29058, 29188, 29408, 29527, 29762]
    output_path = f"/home/dai/GPU-Student-2/Cederic/DataSciPro/study/results/eval_selection/{image_idxs[which_image]}_selec.png"
    user_selection_lst, image_selection_lst = evaluate_Selection(df)
    plot_histogram(image_selection_lst, which_image, output_path)

    ## ------------------ Evaluate the Annotation Task ------------------
    #which_image = 0
    #image_idxs = [27300, 27398, 27591, 27611, 27931, 28113, 28125, 28285, 28355, 28383, 28583, 28782, 28892, 29058, 29408, 29527, 29762]
    #image_paths = [f'/home/dai/GPU-Student-2/Cederic/DataSciPro/data/misclsData_gt1/{idx}_1.0_misclassified.png' for idx in image_idxs]
    #output_path = f"/home/dai/GPU-Student-2/Cederic/DataSciPro/study/results/eval_annotation/{image_idxs[which_image]}_annot.png"
    #evaluate_annotation(df, image_idxs[which_image], image_paths[which_image], output_path)


