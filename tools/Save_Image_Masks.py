from utils_test import labelname2ix , ix2labelname ,get_proper_CT_windowing

print('site_list to save is ' , site_list)
if site_list == [8]:
    name_save = name+'_liver'
else:
    name_save = name+'_allsites'
SAVE_PATH = '/mnt/fast-disk1/mjc/AutoRecist/Outputs/ScaleNAS_Q5_9Slices/%s/CTs_%s/'%(mask_name,name_save)
print( f"Save path is {SAVE_PATH}" )
SHOW_LABEL = True
SHOW_BOX = False
SHOW_MASK = True
# SHOW_UNION_MASK= False
SHOW_MASK_LABEL = True
SHOW_GT_MASK = True
SHOW_GT_LABEL = False




def get_gts(CT_path,df):
    CT_gts = {} #return all the gts in one CT.
    CT_path = CT_path.replace('/Pngs/' , '/Inputs/' )
    df_gts = df[ df['Image File Path'] == CT_path]


    for _,row in df_gts.iterrows():
        location_ix = labelname2ix(row['Location']) 
        reader = WeasisRawFileReader()
        weasis_raw_data = reader.read_weasis_raw_file(row['Contour File Path'])
        slice_list = weasis_raw_data.get_instance_number_array()
        for j, slice_no in enumerate(slice_list):

            mask = weasis_raw_data.get_mask_image_2d(j)
            if np.sum(mask > 0) <= 3 :
                print(' Mask too small only %d points, skip slice_no %d'%(np.sum(mask > 0) , slice_no) )
                continue
            seg, bbox, area = __get_annotation__(mask)
            if slice_no in CT_gts:
                if location_ix in CT_gts[slice_no]:
                    CT_gts[slice_no][location_ix].extend( seg )
                else:
                    CT_gts[slice_no][location_ix]=seg
            else:
                CT_gts[slice_no] = {}
                CT_gts[slice_no][location_ix]=seg

    return CT_gts

    



keys = list(D_CT.keys())
for k in keys:
    # print(k) #/mnt/fast-disk1/mjc/AutoRecist/Pngs/PDS_AUTO_RECIST/METNET0001/D2018_12_03/E2952/CT/S0002_6566
    
    oneCT = remove_single_slice_segms(D_CT[k])
    savepath = os.path.join( SAVE_PATH , '%s'%(convert_name_compact(k)) )
    

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    else:
        print('Skiped' , savepath)
        continue

    print('Image masks are saving to:' , savepath)

    if SHOW_GT_MASK:
        CT_gts = get_gts(k , df)
    
    for s in oneCT:
        aroidb , bboxes , segmentations = oneCT[s]

        image_path = os.path.join(aroidb['image'])
    
        CT_windowing = get_proper_CT_windowing(segmentations)
        if CT_windowing:
            HU1, HU2 = CT_windowing
        else:
            [HU1, HU2 ] = aroidb['windows']
            

        image = load_image(image_path, HU1, HU2)
        height,width = image.shape
    #     plt.imshow(image)
        image = np.dstack((image,image,image))

        if SHOW_GT_MASK and s in CT_gts:
            for j in site_list:
                if j in CT_gts[s]:
                    contours = CT_gts[s][j]
                    colors = [20,30,255]
                    label = ix2labelname(j)
                    for c in contours:
                        c = np.reshape(c,(-1,2))
                        if c.shape[0]:
                            image = cv2.drawContours(image, [np.int64( c )], -1, colors, 1)
                            if SHOW_GT_LABEL:
                                x,y,_,_ =ploy2boxes(c)
                                template = "{}"
                                if len(label)>=9:
                                    label=label[:3]+label[-3:]
                    
                                s = template.format(label)
                                cv2.putText(
                                    image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, colors, 1
                                )  

        
        if SHOW_MASK:
            for j in site_list:
                contours = segmentations[j]
                colors = compute_colors_for_labels(j)
                label = ix2labelname(j)
                for c in contours:
                    c = np.reshape(c,(-1,2))
                    if c.shape[0]:
                        image = cv2.drawContours(image, [np.int64( c )], -1, colors, 1)
                        if SHOW_MASK_LABEL:
                            x,y,_,_ =ploy2boxes(c)
                            template = "{}"
                            if len(label)>=9:
                                label=label[:3]+label[-3:]
                
                            s = template.format(label)
                            cv2.putText(
                                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, colors, 1
                            )  
        

        cv2.imwrite( os.path.join(savepath, convert_name_compact( aroidb['image'] ) ) , image )



