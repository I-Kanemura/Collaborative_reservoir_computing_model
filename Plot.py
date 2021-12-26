import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import ScalarFormatter


class Plot :

	TEXT_elements = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
	IMG_elements = ['r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','g0','g1','g2','g3','g4','g5','g6','g7','g8','g9','b0','b1','b2','b3','b4','b5','b6','b7','b8','b9']
	colors_default = { 0:'black',
								1:'red',
								2:'blue' ,
								3:'green',
								4:'orange',
								5:'olive',
								6:'stateblue',
								7:'darkmagenta',
								8:'saga',
								9:'darkcyan' }

	def __init__( self , Path, SignalLabel, Parameter ) :
		self.path = Path
		self.Label_normal = SignalLabel[0].copy()
		self.Label_mutation = SignalLabel[1].copy()
		self.chunk_size = SignalLabel[2].copy()
		self.delay = Parameter['delay']
		self.stimtime = Parameter['stimtime']
		self.C = len( self.Label_normal ) 
		self.Signal_length = np.sum( self.chunk_size ) * self.stimtime



	def Plot_chunk_area( self, axs, Label ) :
		counter = 0
		for i in range( self.C ) :
			end = counter + self.stimtime*self.chunk_size[ i ]
			if Label[ i ] :
				axs.axvspan( counter, end , facecolor=Plot.colors_default[ Label[ i ] ], alpha=0.3, linewidth=0)
			counter = end

	def Plot_Signal_Text( self, signal, num ) :
		chunk_sum = np.sum( self.chunk_size[ 0 : num ] )
		plt_end = chunk_sum * self.stimtime
		figs = plt.figure( figsize=(18, 9) )
		axs = {}
		for i in range( 26 ) :
			axs[i] = figs.add_subplot( 26, 1, i+1 )
			axs[i].plot( signal[i] , c='black')
			axs[i].tick_params(bottom=False,left=False,right=False,top=False)
			axs[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
			axs[i].set_ylabel(Plot.TEXT_elements[i], fontsize=10)
			axs[i].set_xlim(0, plt_end)
			axs[i].set_ylim(0 , 2.5)
			self.Plot_chunk_area( axs[i], self.Label_normal )
		figs.tight_layout()
		figs.subplots_adjust( hspace=0.3 )
		figs.savefig( self.path+'Text_Signal.pdf' )
		plt.close(figs)


	def Plot_Signal_Image( self, signal, num ) :
		chunk_sum = np.sum( self.chunk_size[ 0 : num ] )
		plt_end = chunk_sum * self.stimtime
		figs = plt.figure( figsize=(18, 9) )
		axs = {}
		for i in range( 30 ) :
			axs[i] = figs.add_subplot( 30, 1, i+1 )
			axs[i].plot( signal[i], c='black')
			axs[i].tick_params(bottom=False,left=False,right=False,top=False)
			axs[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
			axs[i].set_ylabel(Plot.IMG_elements[i], fontsize=10)
			axs[i].set_xlim(0, plt_end)
			axs[i].set_ylim(0 , 2.5)
		return figs , axs


	def Plot_Signal_Image_normal( self, signal, num ) :
		figs ,axs = self.Plot_Signal_Image( signal, num )
		for i in range( 30 ) :
			self.Plot_chunk_area( axs[i], self.Label_normal )
		figs.tight_layout()
		figs.subplots_adjust( hspace=0.3 )
		figs.savefig( self.path+'Image_Signal_normal.pdf' )
		plt.close(figs)


	def Plot_Result( self, ResultRC1, ResultRC2, ResultRC, num , color ) :
		self.color = { 0 : 'black' }
		for i in range( 1 , ( len( color[ 1: ] )+ 1 ) ) :
			self.color[ i ] =Plot. colors_default[ color[ i ] ]
		chunk_sum = np.sum( self.chunk_size[ 0 : num ] )
		plt_end = chunk_sum * self.stimtime
		self.O = ResultRC.shape[ 0 ]
		figs = plt.figure( figsize=(18, 9) )
		axs = {}
		f = 20
		for i in range( 3 ) :
			axs[i] = figs.add_subplot(3, 1, i+1)
		for i in range( self.O ) :
			axs[0].plot( ResultRC1[ i ], c=self.color[i+1], label='node'+str(i+1) )
			axs[1].plot( ResultRC2[ i ], c=self.color[i+1], label='node'+str(i+1) )
			axs[2].plot( ResultRC[ i ], c=self.color[i+1], label='node'+str(i+1) )
		axs[0].set_ylabel( 'RC1 readout z(t)', fontsize=f )
		axs[1].set_ylabel( 'RC2 readout z(t)', fontsize=f )
		axs[2].set_ylabel( 'RC readout o(t)', fontsize=f )
		for i in range( 3 ) :
			axs[i].legend( loc='upper right' )
			axs[i].set_xlabel( 'time [ms]', fontsize=f )
			axs[i].set_xlim( 0, plt_end )
			axs[i].set_ylim(- 1.0, 1.5 )
			axs[i].tick_params( labelsize=f )
			self.Plot_chunk_area( axs[i], self.Label_normal )
		figs.tight_layout()
		figs.subplots_adjust( left=0.1, right=0.95, bottom=0.1, top=0.95 )
		figs.subplots_adjust( hspace=0.6 )
		figs.savefig( self.path+'Test_Result.pdf' )
		plt.close(figs)