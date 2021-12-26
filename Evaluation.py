import numpy as np


class Evaluation :

	dThreshold = 0.1
	Threshold = np.arange( 0,1, dThreshold )
	Threshold_num = len( Threshold )

	def __init__( self, Path, Parameter ) :
		self.path = Path
		self.stimtime = Parameter['stimtime']
		self.chunks = Parameter['chunks']
		self.delay = Parameter['delay']
		self.confusion = None
		self.Accuracy = 0
		self.Result_label = None

	def checkNode( self, signal, label, times ) :
		self.Signal = self.Normalize( signal )
		self.Label = label
		self.Timedata = times
		self.C = len( self.Timedata )
		self.O = self.Signal.shape[0]
		self.Result_threshold = np.zeros( Evaluation.Threshold_num )
		for i in range( Evaluation.Threshold_num ) :
			data = self.applyThreshold( Evaluation.Threshold[i] )
			self.detecteNode( data )
			self.createConfusionMatrix_node()
			self.createChunkLabel()
			self.detecteChunk()
			self.createConfusionMatrix_chunk()
			self.Result_threshold[i] = self.culculate()
			if self.Result_threshold[i] > self.Accuracy :
				self.Accuracy = self.Result_threshold[i]
				self.confusion = self.confusion_buffer_chunk.copy()
				buffer = self.confusion_buffer_node.copy()
				self.Result_label = np.asarray( list( self.Result_label_buffer.values() ) )
		return self.Accuracy , self.Result_label

	def Normalize( self, signal ) :
		stim_max = np.max( signal )
		buffer = signal / stim_max
		return buffer

	def applyThreshold( self, threshold ) :
		flag = self.Signal >= threshold
		return self.Signal * flag

	def detecteNode( self, data ) :
		self.ResultNodedata = np.zeros( self.C )
		start = 0
		for i in range( self.C ) :
			stimulength = self.Timedata[i] * self.stimtime
			end = int( start + stimulength )
			buffer = data[ : , start:end ]
			result = np.sum( buffer, axis=1 )
			node_index = np.argmax( result )
			chunk_num = result [ node_index ]
			result_num = np.sum( buffer )
			random_num = self.C - result_num
			if random_num > chunk_num :
				self.ResultNodedata[i] = 0
			else :
				 self.ResultNodedata[i] = node_index +1
			start = end

	def createChunkLabel( self ) :
		self.Result_label_buffer = { }
		for i in range( self.O + 1 ) :
			buffer = self.confusion_buffer_node[ : , i ] 
			chunk_index = np.argmax( buffer )
			self.Result_label_buffer[ i ] = chunk_index

	def detecteChunk( self ) :
		self.ResultChunkdata = np.zeros( self.C )
		for i in range( self.C ) :
			self.ResultChunkdata[ i ] = self.Result_label_buffer[ self.ResultNodedata[ i ] ]

	def createConfusionMatrix_node( self ) :
		self.confusion_buffer_node = np.zeros( ( self.chunks+1, self.O+1 ) )
		for i in range( self.C ) :
			stim_index = int( self.Label[i] ) 
			node_index = int(  self.ResultNodedata[i] )
			self.confusion_buffer_node[ stim_index , node_index ] += 1

	def createConfusionMatrix_chunk( self ) :
		self.confusion_buffer_chunk = np.zeros( ( self.chunks+1, self.chunks+1 ) )
		for i in range( self.C ) :
			stim_index = int( self.Label[i] ) 
			chunk_index = int(  self.ResultChunkdata[ i ] )
			self.confusion_buffer_chunk[ stim_index , chunk_index ] += 1

	def culculate( self ) :
		buffer = np.diag( self.confusion_buffer_chunk )
		return np.sum( buffer ) / self.C

	def saveResult( self , tag ) :
		with open( self.path+"Accuracy.txt", 'w') as f :
			print( self.Accuracy, file=f)
		np.savetxt(self.path+tag+"_Confusion.csv",  self.confusion , delimiter=",")
		np.savetxt(self.path+tag+"_Color.csv",  self.Result_label , delimiter=",")