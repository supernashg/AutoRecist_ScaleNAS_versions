import os
import json
OLD = '/mnt/fast-disk1/mjc/AutoRecist/Pngs'
NEW = '/raid/cialab/mjc/AutoRecist/Pngs'

def change(coco_images):
    newimages = []
    for image in coco_images:
        file_name = image['file_name']
        image['file_name'] = file_name.replace(OLD, NEW)
        newimages.append(image)
    return newimages

folder = '/raid/cialab/mjc/AutoRecist/Annotations'

namelist = [
    'AMGEN_20020408_20201027_z.json',  
    'AMGEN_PRIME&CUIMC_20210228.json',
    'AMGEN_20020408_20201202.json',     
    'CUIMC_20201027_z.json',
    'AMGEN_PRIME_20201027_z.json',      
    'CUIMC_20201202.json',
    'AMGEN_PRIME_20201202.json',        
    'CUIMC_20210228.json'
]

for name in namelist[:-1]:
    jsonpath = os.path.join(folder,name)
    print('load',jsonpath)
    with open(jsonpath) as f:
        coco = json.load(f)
    print('images len: %d annotations len: %d' %( len(coco['images']) , len(coco['annotations']) ) )

    newimages = change(coco['images'])
    json_data = {"info" : coco['info'],
                 "images" : newimages,
                 "licenses" : coco['licenses'],
                 "annotations" : coco['annotations'] ,
                 "categories" : coco['categories']}
    savename = jsonpath
    with open(savename, "w") as jsonfile:
        json.dump(json_data, jsonfile, sort_keys=True, indent=4)
    print( 'Saved %s'%savename )
        

print(json_data['images'][0])
