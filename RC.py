import numpy as np


class RC :

	def __init__ ( self , Name, Parameter ) :
		self.name = Name
		self.modelType = 'normal'
		self.path = Parameter['path'] + self.name + "_"
		self.I = Parameter[ 'I' ]
		self.N = Parameter[ 'N' ]
		self.O = Parameter[ 'O' ]
		self.S = Parameter[ 'S' ]
		self.g = Parameter[ 'G' ]
		self.gOUT =Parameter[ 'Gout' ]
		self.gFB = Parameter[ 'Gback' ]
		self.p = Parameter[ 'p' ]
		self.dt = Parameter[ 'dt' ]# (ms)
		self.tau = Parameter[ 'tau' ] # (ms)
		self.sigma = Parameter[ 'sigma' ]
		self.saveStatus()
		self.setNode()
		self.setWeight()


	def setNode( self ) :
		self.Xn = 0.5 * np.random.randn( self.N, 1 )
		self.Rn = np.tanh( self.Xn )
		self.Zn = 0.5 * np.random.randn( self.O, 1 )
		self.Synapse = np.random.choice( np.arange( 0, self.N ), self.S, replace=False )
		self.backupNode()


	def backupNode( self ) :
		self.Xn_backup = self.Xn.copy()
		self.Rn_backup = self.Rn.copy()
		self.Zn_backup = self.Zn.copy()
	def loadNode( self ) :
		self.Xn = self.Xn_backup.copy()
		self.Rn = self.Rn_backup.copy()
		self.Zn = self.Zn_backup.copy()


	def setWeight( self  ) :
		self.Win = np.zeros( ( self.N, self.I ) )
		for i in range( self.N ):
			self.Win[ i , np.random.randint( 0, self.I ) ] = np.random.randn()
		scale = 1.0 / np.sqrt( self.p * self.N ) * self.g
		self.W = self.setW( scale )
		self.Wout = np.random.randn( self.O, self.S ) / np.sqrt( self.S ) * self.gOUT
		self.Wback = ( np.random.rand( self.N, self.O ) - 0.5 ) * 2.0 * self.gFB


	def setW( self , scale ) :
		W = np.random.randn( self.N, self.N )
		p_flag = np.random.rand( self.N, self.N )
		# self.p = 1 : full connection
		flag = p_flag <= self.p
		W = W * flag
		return W * scale


	def Update_normal( self, Data ) :
		dx = - self.Xn \
				+ np.dot( self.W, self.Rn ) \
				+ np.dot( self.Wback,  self.Zn ) \
				+ np.dot( self.Win, Data ) 
		self.Xn = self.Xn \
						+ ( self.dt  / self.tau ) * dx \
						+  self.sigma * np.random.randn( self.N, 1 )
		self.Rn = np.tanh( self.Xn )
		self.Zn = np.dot( self.Wout, self.Rn[ self.Synapse, : ] )


	def saveStatus ( self ) :
		with open( self.path+"status.txt", 'w' ) as f :
			print("Name : ",self.name, file=f)
			print("Type : ", self.modelType, file=f)
			print("Input : ",self.I, file=f)
			print("Inner : ",self.N, file=f)
			print("Output : ",self.O, file=f)
			print("Synapse : ",self.S, file=f)
			print("Sparce : ",self.p, file=f)
			print("g : ",self.g, file=f)
			print("g_out : ",self.gOUT, file=f)
			print("g_back : ",self.gFB, file=f)
			print("sigma : ",self.sigma, file=f)
			print("tau : ",self.tau, file=f)


	def saveResult( self ) :
		np.savetxt(self.path+"Synapse.csv", self.Synapse, delimiter=",")