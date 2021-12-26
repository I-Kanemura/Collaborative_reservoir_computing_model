import numpy as np
from PIL import Image




class Signal :

	Text_array = {
						'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,
						'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,
						'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,
						'v':21,'w':22,'x':23,'y':24,'z':25
						}

	def __init__ ( self, Name, Path, Parameter ) :
		self.name = Name
		self.modelType = 'normal'
		self.path = Path +  self.name + "_"
		self.m = Parameter[ 'm' ] 
		self.delay = int( Parameter[ 'delay' ] )
		self.stimtime = Parameter[ 'stimtime' ] 
		self.gain = Parameter[ 'gain' ] 
		self.Chunks =Parameter[ 'chunks' ] 
		self.interval = Parameter[ 'interval' ] 
		self.filter_size = Parameter[ 'filter' ]


	def setTextChunk( self, List ) :
		self.Text_len = 26
		self.Random_elements = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		self.Chunk_Text  = {}
		for i in range( self.Chunks ) :
			self.Chunk_Text[ i + 1 ] = List[ i ]


	def setChunkSize( self, List ) :
		self.Random_size = 26
		self.Chunk_Size ={}
		for i in range( self.Chunks ) :
			self.Chunk_Size[ i + 1 ] = List[ i ]


	def setImageChunk( self, ListPath ) :
		self.Image_len = 30
		self.Chunk_Image = {}
		for i in range( self.Chunks ) :
			self.Chunk_Image[ i + 1 ] = Image.open( ListPath[ i ] )


	def createShape( self ) :
		self.Stimulate_shape = np.zeros( self.stimtime * 2 )
		for i in range( self.stimtime ):
			self.Stimulate_shape[ i ] = ( 1 - np.exp( -( i / 10 ) ) ) * self.gain
			self.Stimulate_shape[ self.stimtime + i ] =  np.exp( -( i / 10 ) ) * self.gain


	def Pooling( self, data, chunk_size ) :
		post_img_size_row = self.Image_len // 3
		post_img_size_cal = self.stimtime * chunk_size
		pri_img_size_row = post_img_size_cal + self.filter_size - 1
		pri_img_size_cal = post_img_size_row + self.filter_size - 1
		data = np.asarray( data.resize( ( pri_img_size_row, pri_img_size_cal ) ) )
		Buffer = np.zeros( ( post_img_size_row, post_img_size_cal, 3 ) )
		for c in range( post_img_size_row ) :
			for r in range( post_img_size_cal ) :
				Buffer[ c, r, 0 ] = np.max( data[ c : c+self.filter_size, r : r+self.filter_size, 0 ] )
				Buffer[ c, r, 1 ] = np.max( data[ c : c+self.filter_size, r : r+self.filter_size, 1 ] )
				Buffer[ c, r, 2 ] = np.max( data[ c : c+self.filter_size, r : r+self.filter_size, 2 ] )
		Result = np.zeros( (self.Image_len, post_img_size_cal ) )
		Result[ 0 : post_img_size_row,                      : ] = Buffer[:,:,0]
		Result[ post_img_size_row*1 : post_img_size_row*2, : ] = Buffer[:,:,1]
		Result[ post_img_size_row*2 :  ,                   : ] = Buffer[:,:,2]
		Result  = Result / 255 * self.gain
		return Result


	def Random_Chunk( self , chunk_size ) :
		Random_text = [''] * chunk_size
		for i in range( chunk_size ) :
			Random_text[ i ] = self.Random_elements[ np.random.randint( 0, self.Random_size ) ]
		filter_size = 9
		signal_row = self.stimtime * chunk_size
		Random_img = np.zeros( (self.Image_len, signal_row) )
		buffer = np.random.rand( self.Image_len, signal_row+filter_size  ) 
		for i in range( signal_row ) :
			temp = buffer[:, i : i + filter_size]
			temp = np.mean( temp, axis=1, keepdims=True )
			Random_img[:,i:i+1] = temp * self.gain
		return Random_text , Random_img


	def createSignal( self, times, tag ) :
		self.tag = tag
		self.C = times
		self.Label = np.zeros( ( 3, self.C ) )
		self.createChunkList( )
		self.Stimulate_len = int( np.sum( self.Label[2] ) * self.stimtime )
		self.createShape( )
		self.Signal_Text_normal = np.zeros( ( self.Text_len , self.Stimulate_len ) )
		self.Signal_Image_normal = np.zeros( ( self.Image_len , self.Stimulate_len ) )
		Image = {}
		for i in range( self.Chunks ) :
			Image[ i + 1 ] = self.Pooling( self.Chunk_Image[ i + 1 ] , self.Chunk_Size[ i + 1 ] )
		target = 0
		for i in range( self.C ) :
			chunk_num = int( self.Label[0][ i ] )
			chunk_size = int( self.Label[2][ i ] )
			if chunk_num == 0 :
				chunk_text_buffer , chunk_image_buffer = self.Random_Chunk( chunk_size )
			else :
				chunk_text_buffer = self.Chunk_Text[ chunk_num ]
				chunk_image_buffer = Image[ chunk_num ]
			for j in range( chunk_size ) :
				input_start = target + j * self.stimtime
				input_end = min( ( input_start + self.stimtime * 2 ) ,  self.Stimulate_len )
				input_row = Signal.Text_array[ chunk_text_buffer[ j ] ]
				self.Signal_Text_normal[ input_row, input_start : input_end ] = self.Stimulate_shape[ 0 : input_end - input_start ]
			input_end = chunk_size*self.stimtime
			self.Signal_Image_normal[ : , target:target+input_end] = chunk_image_buffer
			target = target + self.stimtime * chunk_size
		self.Signal_Image_mutation = self.Signal_Image_normal.copy()
		target = 0
		for i in range( self.C ) :
			chunk_num_normal = int( self.Label[0][ i ] )
			chunk_num_mutation = int( self.Label[1][ i ] )
			chunk_size = int( self.Label[2][ i ] )
			input_end = chunk_size * self.stimtime
			if chunk_num_normal != chunk_num_mutation :
				if chunk_num_mutation == 0 :
					chunk_text_buffer , chunk_image_buffer = self.Random_Chunk( chunk_size )
				else :
					chunk_text_buffer = self.Chunk_Text[ chunk_num ]
					chunk_image_buffer = self.Pooling( self.Chunk_Image[ chunk_num ] , chunk_size )
				self.Signal_Image_mutation[ : , target:target+input_end] = chunk_image_buffer 
			target = target + self.stimtime * chunk_size
		self.Signal_Image_delay = self.Signal_Image_normal.copy()
		self.Signal_Image_delay = np.roll( self.Signal_Image_delay, self.delay )

	def createChunkList( self ) :
		flag = True
		separater = 1 / self.Chunks
		for i in range( self.C ) :
			if flag :
				self.Label[0][i] = 0
				self.Label[1][i] = 0
				self.Label[2][i] = np.random.randint( self.interval[0], self.interval[1] )
				flag = False
			else :
				list_index = ( np.random.rand() // separater ) + 1
				self.Label[0][i] = list_index
				self.Label[1][i] = list_index
				self.Label[2][i]  = self.Chunk_Size[ list_index ]
				flag = True
		self.createMutation()


	def createMutation( self ) :
		for i in range( self.C ) :
			if np.random.rand() < self.m :
				self.Label[1][i] = np.random.randint( 0 , 4 )
			else :
				null = np.random.randint( 0 , 4 )


	def saveStatus( self ) :
		with open(self.path+self.tag+".txt", 'w') as f :
			print("Name : ",self.name, file=f)
			print("Type : ",self.modelType, file=f)
			print("C : ",self.C, file=f)
			print("stimTime : ",self.stimtime, file=f)
			print("mutation : ",self.m, file=f)
			print("delay : ",self.delay, file=f)

	def saveSignal( self ) :
		np.savetxt(self.path+self.tag+"_Label.csv", self.Label, delimiter=",")
