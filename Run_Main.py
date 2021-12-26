import os
os.environ["OMP_NUM_THREADS"] = "2"
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import ScalarFormatter
from PIL import Image
import RC
import Model
import MySignal
import Plot
import Evaluation


simu= 'Main'
root_path = 'Result/'
if not os.path.exists( root_path ):
	os.mkdir( root_path )
root_path = root_path + simu + '/'
if not os.path.exists( root_path ):
	os.mkdir( root_path )
dt_now = datetime.datetime.now()
root_path=  root_path + dt_now.strftime('%Y_%m_%d_%H_%M_%S')+"/"
if not os.path.exists( root_path ):
	os.mkdir( root_path )


ParSignal = {
	'm' : 0.0,
	'delay' : 0,
	'stimtime' : 50,
	'gain' : 2,
	'chunks' : 3,
	'interval' : [3,7],
	'filter' : 3
}
Signal = MySignal.Signal( 'Signal' , root_path , ParSignal ) 
Signal.setChunkSize( [5,5,6] ) 
Signal.setTextChunk([	['a','p','p','l','e'],
									['g','r','a','p','e'], 
									['b','a','n','a','n','a'] ])
Signal.setImageChunk( [	'img/apple.jpg',
										'img/grape.jpg',
										'img/banana.jpg'] )


np.random.seed( 0 )
Signal.createSignal( 2200, 'learning' )
learning_signal_text = Signal.Signal_Text_normal.copy()
learning_signal_image = Signal.Signal_Image_normal.copy()
#learning_signal_image = Signal.Signal_Image_mutation.copy()
#learning_signal_image = Signal.Signal_Image_delay.copy()
learning_label = Signal.Label.copy()


np.random.seed( 10 )
Signal.createSignal( 200, 'test' )
test_signal_text = Signal.Signal_Text_normal.copy()
test_signal_image = Signal.Signal_Image_normal.copy()
#test_signal_image = Signal.Signal_Image_mutation.copy()
#test_signal_image = Signal.Signal_Image_delay.copy()
test_label = Signal.Label.copy()
Signal.saveStatus( )
Signal.saveSignal( )



Parameter = {
						'root_path' : '',
						'path' : root_path,
						'seed' : 0,
						'id' : 0,
						'I' : 0,
						'N' : 1200,
						'O' : 3,
						'S' : 300,
						'RC1' : 26,
						'RC2' : 30,
						'G':  1,
						'Gout' : 1,
						'Gback' : 1,
						'p' : 0.5,
						'dt' : 1,
						'tau' : 5,
						'sigma' : 0.2 ,
						'window' : 15000,
						'span' : 2,
						'alpha' : 100,
						'beta' : 3,
						'gamma' : 0.5,
						'delta' : 0.5
 }
 
 
np.random.seed( Parameter['seed'] )
MODEL = Model.DualRC( 'MODEL', Parameter ) 
MODEL.Learning( learning_signal_text , learning_signal_image )
MODEL.Test( test_signal_text , test_signal_image )


ENGINE = Evaluation.Evaluation( root_path ,ParSignal )
Accuracy , chunkLabel = ENGINE.checkNode( MODEL.TestRC, test_label[0], test_label[2] )
ENGINE.saveResult( '' ) 


plot = Plot.Plot( root_path , Signal.Label, ParSignal )
plot.Plot_Signal_Text( Signal.Signal_Text_normal , 20 )
plot.Plot_Signal_Image_normal( Signal.Signal_Image_normal , 20 )
plot.Plot_Result( MODEL.TestRC1_result, MODEL.TestRC2_result, MODEL.TestRC, 20 , chunkLabel )