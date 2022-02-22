import os
import shutil
file = os.listdir('../../../Andrew_ng/seed/ExtractedFeatures/')
file.sort()
for i in range(0, 45, 3):
    shutil.copy('../../../Andrew_ng/seed/ExtractedFeatures/'+file[i], './SEED/1/')
    shutil.copy('../../../Andrew_ng/seed/ExtractedFeatures/'+file[i+1], './SEED/2/')
    shutil.copy('../../../Andrew_ng/seed/ExtractedFeatures/'+file[i+2], './SEED/3/')
