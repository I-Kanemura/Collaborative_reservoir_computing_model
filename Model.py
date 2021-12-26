import numpy as np
import RC


class DualRC :

	def __init__( self, Name, Parameter ) :
		self.seed = Parameter['seed']
		self.prosess_id = Parameter['id']
		self.name = Name
		self.modelType = 'normal'
		self.path =  Parameter['path'] +  self.name + "_"
		Parameter['I'] = Parameter['RC1']
		self.RC1 = RC.RC( 'RC1' , Parameter )
		Parameter['I'] = Parameter['RC2']
		self.RC2 = RC.RC( 'RC2' , Parameter )
		self.window = Parameter[ 'window' ]
		self.span = Parameter[ 'span' ]
		self.alpha = Parameter[ 'alpha' ]
		self.beta = Parameter[ 'beta' ]
		self.gamma = Parameter[ 'gamma' ]
		self.delta = Parameter[ 'delta' ]
		self.P1 = ( 1.0 / self.alpha ) * np.identity( self.RC1.S )
		self.P2 = ( 1.0 / self.alpha ) * np.identity( self.RC2.S )
		self.saveStatus() 


	def Learning( self, signalRC1, signalRC2 ) :
		times = signalRC1.shape[1]
		LearningRC1_result = np.zeros( ( self.RC1.O, self.window ) )
		LearningRC2_result = np.zeros( ( self.RC2.O, self.window ) )
		self.RC1.loadNode()
		self.RC2.loadNode()
		for i in range( times ):
			self.RC1.Update_normal( signalRC1[ : , i:i+1 ] )
			self.RC2.Update_normal( signalRC2[ : , i:i+1 ] )
			LearningRC1_result = np.roll( LearningRC1_result, -1, axis=1)
			LearningRC2_result = np.roll( LearningRC2_result, -1, axis=1)
			LearningRC1_result[ : , -2:-1 ] = self.RC1.Zn
			LearningRC2_result[ : , -2:-1 ] = self.RC2.Zn
			if self.window < i and i % self.span == 0 :
				self.RC1_zhat = self.Normalize( self.RC1.Zn, LearningRC1_result )
				self.RC2_zhat = self.Normalize( self.RC2.Zn, LearningRC2_result )
				self.RC1_f = self.createTeaching( self.RC2_zhat )
				self.RC2_f = self.createTeaching( self.RC1_zhat )
				self.RC1_error = self.RC1.Zn - self.RC1_f
				self.RC2_error = self.RC2.Zn - self.RC2_f
				self.P1, self.RC1.Wout = self.Update( self.P1, self.RC1_error, self.RC1.Rn[ self.RC1.Synapse, : ] ,self.RC1.Wout )
				self.P2, self.RC2.Wout = self.Update( self.P2, self.RC2_error, self.RC2.Rn[ self.RC2.Synapse, : ] ,self.RC2.Wout )
		self.RC1.backupNode()
		self.RC2.backupNode()
		self.saveResult_Learning()


	def Update( self, P, error, synapse , Wout ) :
		Pr = np.dot( P, synapse  )
		PrrP=  np.outer( Pr, Pr )
		rPr = np.dot( synapse.T, Pr )
		rPr_1 = 1.0 / ( 1.0 + rPr )
		dP = PrrP * rPr_1
		P = P - dP
		dwRC = np.dot( error, Pr.T )
		dwRC =  dwRC * rPr_1
		Wout = Wout - dwRC
		return P , Wout 

	def Normalize( self , data , buffer ) :
		mean = np.mean( buffer, axis=1, keepdims=True )
		std = np.std( buffer, axis=1, keepdims=True )
		return ( data - mean ) / std

	def createTeaching( self, norm ) :
		sumbuffer = np.sum(  norm , keepdims=True )
		sumbuffer = self.gamma * ( sumbuffer - norm )
		f = ( 1.0 / self.beta ) * ( norm - sumbuffer )
		f = np.tanh( f )
		for i in range( f.shape[0] ) :
			if f [ i , ] <= 0 :
				f [ i , ] = 0.0
		return f

	def Test( self, signalRC1, signalRC2  ) :
		times = signalRC1.shape[1]
		self.TestRC1_result = np.zeros( ( self.RC1.O, times ) )
		self.TestRC2_result = np.zeros( ( self.RC2.O, times ) )
		self.TestRC = np.zeros( ( self.RC1.O, times ) )
		self.TestRC1_acticity = np.zeros( ( self.RC1.N, times ) )
		self.TestRC2_acticity = np.zeros( ( self.RC2.N, times ) )
		for i in range( times ):
			self.RC1.Update_normal( signalRC1[ : , i:i+1 ] )
			self.RC2.Update_normal( signalRC2[ : , i:i+1 ] )
			self.TestRC1_result[ : , i:i+1 ] = self.RC1.Zn
			self.TestRC2_result[ : , i:i+1 ] = self.RC2.Zn
			self.TestRC[ : , i:i+1 ] = self.delta * ( self.RC1.Zn + self.RC2.Zn )
			self.TestRC1_acticity[ : , i:i+1 ] = self.RC1.Xn
			self.TestRC2_acticity[ : , i:i+1 ] = self.RC2.Xn
		self.saveResult_Test()

	def saveStatus( self ) :
		with open(self.path+"status.txt", 'w') as f :
			print("path : ", self.path, file=f)
			print("seed : ", self.seed, file=f)
			print("prosess id : ", self.prosess_id, file=f)
			print("Name : ",self.name, file=f)
			print("Type : ", self.modelType, file=f)
			print("Alpha : ", self.alpha, file=f)
			print("Beta : ", self.beta, file=f)
			print("Gamma : ", self.gamma, file=f)

	def saveResult_Learning( self ) :
		self.RC1.saveResult( )
		self.RC2.saveResult( )

	def saveResult_Test( self ) :
		np.savetxt(self.path+"TestRC1.csv", self.TestRC1_result, delimiter=",")
		np.savetxt(self.path+"TestRC2.csv", self.TestRC2_result, delimiter=",")
		np.savetxt(self.path+"TestRC1_activity.csv", self.TestRC1_acticity , delimiter=",")
		np.savetxt(self.path+"TestRC2_activity.csv", self.TestRC2_acticity , delimiter=",")
		np.savetxt(self.path+"TestRC.csv", self.TestRC , delimiter=",")