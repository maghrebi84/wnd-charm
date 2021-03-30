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

pychrm_test_dir = dirname( realpath( __file__ ) )
wndchrm_test_dir = join( dirname( pychrm_test_dir ), 'wndchrm_tests' )
img_filename = "Intensity.tif"
label_filename = "Label.tif"
Sig_reference_filename = "Reference.sig"
img_filepath = pychrm_test_dir + sep + img_filename
label_filepath = pychrm_test_dir + sep + label_filename
Sig_reference_filepath = pychrm_test_dir + sep + Sig_reference_filename

class TestFeatureCalculation(unittest.TestCase):

    def test_ProfileLargeFeatureSet( self ):

        with open( Sig_reference_filepath ) as infile:
            for i, line in enumerate( infile ):
                if i is 0: 
                    name = line.rstrip("\n").split(",")
                if i is 1:
                    values = line.rstrip("\n").split(",")
        
        # Read Label Image and convert it to double *  
        from PIL import Image
        im = Image.open(label_filepath) 
        imarray = np.array(im) 
        height=imarray.shape[0]
        width=imarray.shape[1]
        flattenimarray=imarray.flatten()
        uniqueClasses=np.unique(flattenimarray)
        del_arr = np.delete(uniqueClasses, np.where(uniqueClasses == [0]), axis=0) #delete Label 0
        classID=del_arr.tolist()[0]

        labeledMatrix = wndcharm.doubleArray(width*height)
        i=0
        for element in np.nditer(flattenimarray): 
            labeledMatrix[i] = float(element)
            i=i+1

        del imarray, flattenimarray, uniqueClasses, del_arr

        #Create a ImageMatrix Object and fill it with ROI pixels 
        from PyImageMatrix import PyImageMatrix
        original_px_plane = PyImageMatrix()
        original_px_plane.BoundingBoxFlag=True #Otherwise, Tamura directionality will give an error
        retval = original_px_plane.OpenImage2(label_filepath,width, height, labeledMatrix ,classID)

        if 1 != retval:
            errmsg = 'Could not build a wndchrm.PyImageMatrix from {0}, check the path.'
            raise ValueError( errmsg.format( self.label_filepath ) )     

        #comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSet(True)
        #comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSetLong(True)

        ImageTransformationName='Original'
        FeatureAlgorithmName='PixelStatistics'
        comp_plan = wndcharm.StdFeatureComputationPlans.getFeatureSetbyName(ImageTransformationName,FeatureAlgorithmName)
        #wndcharm.FeatureComputationPlan( name )

        # pre-allocate space where the features will be stored (C++ std::vector<double>)
        tmp_vec = wndcharm.DoubleVector( comp_plan.n_features )

        # Get an executor for this plan and run it
        plan_exec = wndcharm.FeatureComputationPlanExecutor(comp_plan)
        plan_exec.run( original_px_plane, tmp_vec, 0 )

        # get the feature names from the plan
        comp_names = [ comp_plan.getFeatureNameByIndex(i) for i in xrange( comp_plan.n_features ) ]

        # convert std::vector<double> to native python list of floats
        comp_vals = np.array( list( tmp_vec ) )
        np.savetxt('test1.txt', comp_vals, fmt='%g')

        path = '/home/maghrebim2/Work/WND-CHARM/ROI/PR2_Review/wnd-charm/tests/pywndcharm_tests/alaki.sig'

        with open( path, "w" ) as out: 
            for element in np.nditer(comp_names):            
                #out.write("{0:0.8g}\t{1}\n".format( element,element ) )
                alaki=2

        alaki=255

if __name__ == '__main__':
    unittest.main()
