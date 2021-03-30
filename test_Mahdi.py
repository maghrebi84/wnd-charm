import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
import numpy as np
import wndcharm
from wndcharm.FeatureVector import FeatureVector, GenerateFeatureComputationPlan, \
        IncompleteFeatureSetError
from wndcharm.utils import compare

from os.path import dirname, sep, realpath, join, abspath, splitext, basename

sys.path.insert(1, '/home/maghrebim2/Work/WND-CHARM/ROI/PR2_Review/wnd-charm/wndcharm')
'''
import os.path
fullpath='/home/maghrebim2/Work/WND-CHARM/ROI/PR2_Review/wnd-charm/wndcharm/PyImageMatrix.py'
filename='PyImageMatrix.py'
path, filename = os.path.split(fullpath)
filename, ext = os.path.splitext(filename)
sys.path.append(path)
module = __import__(filename)
reload(module) # Might be out of date
del sys.path[-1]
#return module
'''


pychrm_test_dir = dirname( realpath( __file__ ) )
wndchrm_test_dir = join( dirname( pychrm_test_dir ), 'wndchrm_tests' )
test_dir = wndchrm_test_dir

img_filename = "Intensity.tif"
orig_img_filepath = pychrm_test_dir + sep + img_filename

#input_image_path = sourcedir + sep + img_filename
class TestFeatureCalculation( unittest.TestCase ):

    def test_ProfileLargeFeatureSet( self ):
        kwargs = {}
        kwargs[ 'name' ] = img_filename
        kwargs[ 'source_filepath' ] = orig_img_filepath
        #kwargs[ 'feature_names' ] = fw.feature_names
        #kwargs[ 'feature_computation_plan' ] = comp_plan
        kwargs[ 'long' ] = True
        kwargs[ 'x' ] = 0
        kwargs[ 'y' ] = 0

        ROI_width = 231
        ROI_height = 20
        kwargs[ 'w' ] = ROI_width
        kwargs[ 'h' ] = ROI_height

        kwargs[ 'sample_group_id' ] = 0
       # top_left_tile_feats = FeatureVector( **kwargs ).GenerateFeatures( quiet=False, write_to_disk=False)

        Sig_reference_filename = "Reference.sig"
        Sig_reference_filepath = pychrm_test_dir + sep + Sig_reference_filename
        with open( Sig_reference_filepath ) as infile:
            #firstline = infile.readline()
            #m = re.match( r'^(\S+)\s*(\S+)?$', firstline           
            #class_id, input_fs_version = m.group( 1, 2 )
            #input_fs_major_ver, input_fs_minor_ver = input_fs_version.split('.')                
            #orig_source_tiff_path = infile.readline()
          
          #  vec_len = 5
          # values = np.zeros( vec_len )
           # names = [None] * vec_len
            for i, line in enumerate( infile ):
                if i is 0: 
                    name = line.rstrip("\n").split(",")
                if i is 1:
                    values = line.rstrip("\n").split(",")
              #  values[i] = float( val )

        comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSet(True)

        from PyImageMatrix import PyImageMatrix
        original_px_plane = PyImageMatrix()
        reference_sigs_Intensity = '/home/maghrebim2/Work/WND-CHARM/ROI/PR2_Review/wnd-charm/tests/pywndcharm_tests/Intensity.tif'
        reference_sigs_Label = '/home/maghrebim2/Work/WND-CHARM/ROI/PR2_Review/wnd-charm/tests/pywndcharm_tests/Label.tif'

        from PIL import Image
        im = Image.open(reference_sigs_Label) 
        imarray = np.array(im) 
        height=imarray.shape[0]
        width=imarray.shape[1]
        flattenimarray=imarray.flatten()
        classID=255
        labeledMatrix = wndcharm.doubleArray(width*height)
        i=0
        for element in np.nditer(flattenimarray): 
            labeledMatrix[i] = float(element)
            i=i+1
            #for i in range(16):
             #   labledMatrix[i]=i

        #kwargs[ 'long' ] = True
        original_px_plane.BoundingBoxFlag=True
        retval = original_px_plane.OpenImage2(reference_sigs_Intensity,width, height, labeledMatrix ,classID)

        if 1 != retval:
            errmsg = 'Could not build a wndchrm.PyImageMatrix from {0}, check the path.'
            raise ValueError( errmsg.format( self.source_filepath ) )     

        # pre-allocate space where the features will be stored (C++ std::vector<double>)
        tmp_vec = wndcharm.DoubleVector( comp_plan.n_features )

        # Get an executor for this plan and run it
        plan_exec = wndcharm.FeatureComputationPlanExecutor( comp_plan )
        plan_exec.run( original_px_plane, tmp_vec, 0 )

        # get the feature names from the plan
        comp_names = [ comp_plan.getFeatureNameByIndex(i) for i in xrange( comp_plan.n_features ) ]

        # convert std::vector<double> to native python list of floats
        comp_vals = np.array( list( tmp_vec ) )
        np.savetxt('test1.txt', comp_vals, fmt='%g')

        path = '/home/maghrebim2/Work/WND-CHARM/ROI/PR2_Review/wnd-charm/tests/pywndcharm_tests/alaki.sig'

        with open( path, "w" ) as out: 
            for element in np.nditer(comp_names):            
                #out.write( "{0:0.8g}\t{1}\n".format( element,element ) )
                alaki=2

        alaki=255

if __name__ == '__main__':
    unittest.main()
