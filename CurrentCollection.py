'''
Created on Dec 7, 2012

@author: sieck
'''
#from ThreadedMagCalc import verbose, ThreadedMagCalc as MagCalcFunc
from MultiProcMagCalc import verbose, MultiProcMagCalc as MagCalcFunc, n_procs
from math import pi, log, atan2, cos, sin, sqrt
from numpy import array, dot, cross,vstack
from numpy.linalg import solve, norm


xml_param_format='    <ParamWithValue>\n      <name>{0}</name>\n      <typeCode>{1}</typeCode>\n      <value>{2:.4f} {1}</value>\n      <comment />\n      <isKey>false</isKey>\n      </ParamWithValue>\n'
class xml_blob():
   def __init__(self,content):
      self.header='''<?xml version="1.0" encoding="utf-8"?>
<ParamWithValueList xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <version>20080502</version>
  <parameterTypes>
    <ParamType>
      <typeName>Inventor</typeName>
      <typeCode>0</typeCode>
    </ParamType>
    <ParamType>
      <typeName>String</typeName>
      <typeCode>1</typeCode>
    </ParamType>
    <ParamType>
      <typeName>Boolean</typeName>
      <typeCode>2</typeCode>
    </ParamType>
  </parameterTypes>
  <parameters>
'''   
      self.content=content
      self.footer='  </parameters>\n</ParamWithValueList>'
   def __str__(self):
      return self.header+self.content+self.footer
   def __add__(self, other):
      try:
         return xml_blob(self.content+other.content)
      except:
         return xml_blob(self.content+other)
   def __radd__(self, other):
      try:
         return xml_blob(other.content+self.content)
      except:
         return xml_blob(other+self.content)
   def write(self,fid):
      fid.write(str(self))


class CurrentCollection():
   def __init__(self,iterable=None):
      if iterable is None:
         self.items=[]
      else:
         self.items=iterable
   
   def set_I(self, I):
      for thing in self:
         thing.set_I(I)
         
   def move(self, displacement):
      for thing in self:
         thing.move(displacement)
   
   def discretize_current(self):
      IdL=[]
      points=[]
      for thing in self:
         idl,pts=thing.discretize_current()
         IdL.append(idl)
         points.append(pts)
      return vstack(IdL),vstack(points)
         
   def force(self,B):
      IdL,points=self.discretize_current()
      F=cross(IdL, B(points)).sum(axis=0)
      return F
   
   def torque(self,B,centerpoint):   
      IdL,points=self.discretize_current()
      F=cross(IdL, B(points))
      dR=points-centerpoint
      T=cross(dR,F).sum(axis=0)
      return T
         
   def __len__(self):
      return len(self.items)
   def __iter__(self):
      return self.items.__iter__()
   def __getitem__(self,item):
      if isinstance(item,slice):
         return CurrentCollection(self.items.__getitem__(item))
      else:
         return self.items.__getitem__(item)
   def __add__(self,other):
      if isinstance(other,CurrentCollection):
         return CurrentCollection(self.items+other.items)
      else:
         return CurrentCollection(self.items+other)
   def __radd__(self,other):
      if isinstance(other,CurrentCollection):
         return CurrentCollection(other.items+self.items)
      else:
         return CurrentCollection(other+self.items)

   def append(self,item):
      self.items.append(item)
   def extend(self,iterable):
      self.items.extend(iterable)
   def insert(self,idx,item):
      self.items.insert(idx,item)
   def remove(self,item):
      self.items.remove(item)
   def pop(self,idx=-1):
      self.items.pop(idx)
   def flat(self):
      flatlist=[]
      for thing in self:
         try:
            flatlist.extend(thing.flat())
         except:
            flatlist.append(thing)
      return CurrentCollection(flatlist)
         
      
   def B(self,R):
      return MagCalcFunc(self.flat(),'B',R)
   def A(self,R):
      return MagCalcFunc(self.flat(),'A',R)

   def render(self, **keywords):
      for thing in self:
#          print(thing.name)
         thing.render(**keywords)

class CoilPack(CurrentCollection):
         
   def I(self):
      return(self[0].I)
   def render(self,n_points=None,**keywords):
      import mayavi.mlab as MLab
      from numpy import array,vstack,matrix,empty,empty_like,linspace,pi,cos,sin,amax,amin,sqrt,float64
      from scipy.spatial import ConvexHull
      
      if n_points is None:
         try:
            n_points=self.n_points
         except:
            n_points=50
      
      self[0].update()
      rotmtx=self[0].rotmtx
      r=[]
      z=[]
      for thing in self:
         try:
            render_type=thing.wire['render_type']
         except:
            render_type='round'
         if render_type == 'square':
            Xlocal=rotmtx*matrix(thing.r0-self[0].r0).transpose()
            r.extend(thing.R    +thing.wire['width']/2*array([-1,-1,1,1]))
            z.extend(Xlocal[2,0]+thing.wire['width']/2*array([-1,1,1,-1]))
         else:
            try:
               rminor=thing.wire['diam']/2
            except:
               rminor=thing.r_minor
            if rminor is None:
               rminor=thing.R/1000
            Xlocal=rotmtx*matrix(thing.r0-self[0].r0).transpose()
            r.extend(thing.R    +rminor*array([-1,0,1,0]))
            z.extend(Xlocal[2,0]+rminor*array([0,1,0,-1]))
      hull=ConvexHull(vstack((r,z)).transpose())
      vertex_index=list(hull.simplices[0,:])
      next_facet=hull.neighbors[0,0]
      while True:
         next_indices=hull.simplices[next_facet,:]
         if next_indices[0] == vertex_index[-1]: # this vertex was counted with the previous facet
            vertex_index.append(next_indices[1])
            next_facet=hull.neighbors[next_facet,0]
         else: #the other vertex belonged to the previous facet
            vertex_index.append(next_indices[0])
            next_facet=hull.neighbors[next_facet,1]
         if next_facet == 0:
            break
      r=hull.points[vertex_index,0]
      z=hull.points[vertex_index,1]
      
      Xloc=empty((len(r),n_points),dtype=float64)
      Yloc=empty_like(Xloc)
      Zloc=empty_like(Xloc)
      theta=linspace(0,2*pi,n_points)
      for i in range(len(r)):
         Xloc[i,:]=r[i]*cos(theta)
         Yloc[i,:]=r[i]*sin(theta)
         Zloc[i,:] = z[i]
      Rglobal=rotmtx.transpose()*matrix(vstack((Xloc.flatten(),Yloc.flatten(),Zloc.flatten())))
      Xglobal=array(Rglobal[0,:].reshape(Xloc.shape)+self[0].r0[0])
      Yglobal=array(Rglobal[1,:].reshape(Xloc.shape)+self[0].r0[1])
      Zglobal=array(Rglobal[2,:].reshape(Xloc.shape)+self[0].r0[2])
      if 'name' not in keywords:
         try:
            keywords['name']=self.name
         except:
            keywords['name']="CoilPack"
      if 'color' not in keywords:
         try:
            keywords['color']=self.color
         except:
            keywords['color']=(0.4,0.4,0.4)
      if 'opacity' not in keywords:
         try:
            keywords['opacity']=self.opacity
         except:
            keywords['opacity']=1.0
      self.mlab_object=MLab.mesh(Xglobal,Yglobal,Zglobal, **keywords)
      return self.mlab_object 


class square_pack(CoilPack):
   '''
   TODO: this should be a specific case of an arbitrary cross-section winder, so it should just generate the square cross-section and then call super().__init__
   '''
   def move(self,displacement):
      self.r0 += displacement
      for item in self:
         item.move(displacement)

   def _calc_theta_phi_corners(self):
      # Positive phi goes from ID,-face,OD,+face, returning -pi or pi for the ID/+face edge)
      # _phi_centers are the angular positions of minimum minor radius at [ID,-face,OD,+face] 
      # this means that the coil normal is parallel to _phi_centers[-1] and the outward face to _phi_centers[2] 
      self._Rmajor=(self.OD+self.ID)/4
      rr=(self.OD-self.ID)/4
      hh=self.h/2
      angle=2*atan2(hh,rr)
      self._phi_corners=array([-pi+angle, 0, angle,99]) # last one is a catch-all
      self._phi_centers=-pi+array([0,1,2,3])*pi/2+angle/2
      self._r_centers=[rr,hh,rr,hh]
      if (self.normal[0]==self.normal[1] and self.normal[1]==self.normal[2]):
         self._ax2=array([1,0,0.0])
      else:
         self._ax2=self.normal[(1,2,0),]
      self._ax2=cross(self.normal,self._ax2)
      self._ax2 /= norm(self._ax2)
      self._ax3 = cross(self.normal,self._ax2)
   def _surface_params_from_X(self,X): 
      ''' returns (toroidal angle, poloidal angle, minor radius) of a point on this coil's surface in the direction of the arbitrary point X '''
      t,p,_=self._toroidal_params_from_X(X)
      for _corner,_center,_r in zip(self._phi_corners, self._phi_centers, self._r_centers):
         if (p <= _corner):
            r = _r / cos(p - _center)
            break
      return t,p,r
   def _toroidal_params_from_X(self,R):
      ''' returns (toroidal angle, poloidal angle, minor radius) of an arbitrary point X ''' 
      X0=R-self.r0 # vector from coil center
      Z=dot(X0,self.normal) # axial coordinate
      X1=X0-Z*self.normal # cylindrical-radial vector
      RR=norm(X1) # major (cylindrical) radius
      xx=dot(X1,self._ax2)
      yy=dot(X1,self._ax3) 
      theta=atan2(yy,xx)
      r=sqrt(Z**2+(RR-self._Rmajor)**2) # minor radius
      phi=atan2(Z,RR-self._Rmajor)+self._phi_centers[2] # adding _phi_centers[2] can take us out of the range [-pi,pi]
      phi= (phi+pi)%(2*pi)-pi # shifts up by pi, wraps into the range [0,2pi) and then shifts back down by pi, so we're in the range [-pi,pi)
      return theta,phi,r

   def _X_from_toroidal_params(self,theta,phi,r): 
      ''' Coordinates (global system) of a point given its toroidal angle, poloidal angle, minor radius ''' 
      toroidal_vector=self._ax2*cos(theta) + self._ax3*sin(theta) # points toward the major radius point from the center point
      phi2=phi-self._phi_centers[-1] # we've baked in a weird offset. Maybe I'll make this the zero-phi orientation at some later time
      poloidal_vector=self.normal*cos(phi2)-toroidal_vector*sin(phi2) # points toward X from the major radius point
      X=self.r0.copy() # center point
      X += self._Rmajor*toroidal_vector # major radius point
      X += r*poloidal_vector # we've arrived
      return X

   def collide(self,other,clearance=0.0,report=False):
      # check #1, easiest: Do the two axes ever come close enough?
      linkvec=cross(other.normal,self.normal)
      linkvec /= norm(linkvec)
      RHS = other.r0 - self.r0
      LHS = array([self.normal, -other.normal, linkvec]).T
      solvec=solve(LHS, RHS)
      if (abs(solvec[2]) > (self.OD+other.OD)/2+clearance):
         return False # axes never get close enough
      self._calc_theta_phi_corners()
      other._calc_theta_phi_corners()
      def minifunc(X):
         _,_,R1=self._toroidal_params_from_X(X) # minor-radius distance to X from this coil
         _,_,r1=self._surface_params_from_X(X) # minor-radius distance to this coil's surface in the direction of X
         _,_,R2=other._toroidal_params_from_X(X) # minor-radius distance to X from the other coil
         _,_,r2=other._surface_params_from_X(X) # minor-radius distance to the other coil's surface in the direction of X
         return (R1**2+R2**2)-(r1**2+r2**2)
      from scipy.optimize import minimize
      if clearance>0:
         xatol=clearance/20
      else:
         xatol=self.AWG_dict['size']/10
      result=minimize(minifunc, (self.r0+other.r0)/2,method = 'Nelder-Mead',options={'xatol':xatol})
      points=[c._X_from_toroidal_params(*(c._surface_params_from_X(result.x))) for c in [self,other]]
      dX=norm(points[1]-points[0])
      collision=(result.fun < 0) or (dX < clearance)
      if report:
         # xxx=result.x
         # print('Minimized point: {} [m]; {} [in]'.format(xxx, xxx/.0254))
         # xml=''
         # for i,ax in enumerate('xyz'):
            # xml+=xml_param_format.format('mini_{}'.format(ax),'m',xxx[i])
         # for c in [self,other]:
            # xxx=c._X_from_toroidal_params(*(c._surface_params_from_X(result.x)))
            # for i,ax in enumerate('xyz'):
               # xml+=xml_param_format.format('{}_surf_{}'.format(c.name,ax),'m',xxx[i])
         if collision:
            if (result.fun<=0):
               print('Overlap between {} and {} is {} m (INTERFERENCE)'.format(self.name,other.name,dX))
            else:
               print('Clearance between {} and {} is {} m (INSUFFICIENT CLEARANCE)'.format(self.name,other.name,dX))
         else:
            print('Clearance between {} and {} is {} m (ok)'.format(self.name,other.name,dX))
         # with open(r'M:\Projects\Concordia Tri-Axial\Formless v2 6coil\probe.xml','w') as fid:
            # xml_blob(xml).write(fid)
      if result.success:
         return collision
      else:
         print(result.message)

   def approx_inductance(self):
      b=(self.OD+self.ID)/4 #major radius
      a=(self.OD-self.ID)/4 #minor radius
      L1=pi*4e-7*b*(log(8*b/a)-7/4.0) #single-turn radius
      return L1*len(self.items)**2 #scale to number of turns
   def calculate_mass(self):
      self.mass=0.0
      self.length=0.0
      for coil in self:
         this_L=2*pi*coil.R
         self.length += this_L
         self.mass += this_L * self.AWG_dict['kg/m']
      return self.mass
   def report(self):
      try:
         print(self.name)
      except:
         print('Unnamed coil')
      total_turns=len(self.items)
      print('Using {} gage {} wire'.format(self.AWG_dict['gage'],self.AWG_dict['type']))
      print('{} axial windings, {} radial layers ({} total)'.format(self.n_axial,self.n_radial,total_turns))
      print('{:.2g} A current ({:.2g} A*turns)'.format(self[0].I, total_turns*self[0].I))
      m=self.calculate_mass()
      self.area=self.h*(self.OD-self.ID)/2
      self.volume=self.h*pi*(self.OD**2-self.ID**2)/4
      self.resistance = self.AWG_dict['ohm/m']*self.length
      volts=self.resistance*self[0].I
      print('{:.2f} m wire length, {:.2f} kg wire mass'.format(self.length, m))
      copper_vol=self.AWG_dict['area']*self.length
      print('Estimated copper fraction is {:.1%} ({:.3g} cc other materials)'.format(copper_vol/self.volume, (self.volume-copper_vol)*1e6))
      power=volts*self[0].I
      print('{:.2f} ohms, {:.2f} V, {:.2f} W'.format(self.resistance, volts, power))
      try:
         print('ID = {:.2f} cm'.format(self.ID*100))
      except:
         pass
      try:
         print('OD = {:.2f} cm'.format(self.OD*100))
      except:
         pass
      try:
         print('h = {:.2f} cm'.format(self.h*100))
      except:
         pass
      try:
         # single_turn_conductance=sum([1/(2*pi*turn.R) for turn in self])/self.AWG_dict['ohm/m']
         # equivalent_conductivity=single_turn_conductance*2*pi/self.h/log(self.OD/self.ID)
         # print('Equivalent resistivity for COMSOL = {:.3g} ohm-m'.format(1/equivalent_conductivity))
         # print('Pack area for COMSOL = {:.3g} m^2'.format(self.area))
         # print('Resistance/length for COMSOL = {:.3g} ohm/m'.format(self.AWG_dict['ohm/m']))
         # print('Equivalent mass density for COMSOL = {:.0f} kg/m^3'.format(self.mass/volume))
         # print('Equivalent current density for COMSOL = {:.0f} A/m^2'.format(self[0].I*total_turns/area))
         coeff=34 # W/m^2/K
         dT=40
         ai=pi*self.ID*self.h
         print('ID area = {:.1f} cm^2, water-cooling capability about {:.0f} W'.format(ai*100**2,ai*coeff*dT))
         af=pi*(self.OD**2-self.ID**2)/4
         print('face area = {:.1f} cm^2, water-cooling capability about {:.0f} W'.format(af*100**2,af*coeff*dT))
         ao=pi*self.OD*self.h
         print('OD area = {:.1f} cm^2, water-cooling capability about {:.0f} W'.format(ao*100**2,ao*coeff*dT))
         a_total=ai+2*af+ao
         print('Total water-cooling capability about {:.0f} W'.format(a_total*coeff*dT))
         print('simplistic dT was {} K, but power requirement says dT={:.1f} K'.format(dT,power/coeff/a_total))
      except:
         pass
   
   def NA(self):
      NA=0
      for turn in self.flat():
         NA += pi*turn.R**2
      return NA
   
   def simplify(self,simplify=1):
      raise NotImplementedError()
   
   def xml(self):
      xml = xml_param_format.format(self.name+'_ID','cm',self.ID*100)    
      xml += xml_param_format.format(self.name+'_OD','cm',self.OD*100)    
      xml += xml_param_format.format(self.name+'_h','cm',self.h*100)    
      xml += xml_param_format.format(self.name+'_x0','cm',self.r0[0]*100)    
      xml += xml_param_format.format(self.name+'_y0','cm',self.r0[1]*100)    
      xml += xml_param_format.format(self.name+'_z0','cm',self.r0[2]*100)    
      return xml_blob(xml)
      
   
def square_winder(AWG_spec,center_point,coil_normal,ID,OD,h, I=1.0, simplify=None, turns_limit=None):
   '''
    TODO: make part of the square_pack class __init__, then call super().__init__(args...) to finish the init
   ''' 
   from numpy import floor, linspace
   from MagCoil import MagCoil
   try:
      _=AWG_spec['type']
      AWG_dict=AWG_spec
   except: # AWG_spec probably wasn't a dict; assume square wire
      from AWG import AWG_square
      AWG_dict=AWG_square[AWG_spec]
      
   nr=int(floor((OD-ID)/2/AWG_dict['size']))
   nh=int(floor(h/AWG_dict['size']))
   coil=square_pack([])
   coil.normal=coil_normal/sqrt(sum(coil_normal**2)) # make sure it's actually normalized
   coil.n_turns=0
   coil.n_radial=0
   for i in range(nr):
      try:
         if (coil.n_turns >= turns_limit):
            break
      except:
         pass
      coil.n_radial += 1
      layer_R= ID/2 + (i+0.5)*AWG_dict['size']
      for y in linspace(-(h-AWG_dict['size'])/2,(h-AWG_dict['size'])/2,nh):
         try:
            if (coil.n_turns >= turns_limit):
               break
         except:
            pass
         coil.n_turns += 1
         coil.append(MagCoil(y*coil.normal+center_point,
                             coil.normal, R=layer_R, I=I, wire=AWG_dict))
   coil.n_axial=nh
   coil.AWG_dict=AWG_dict
   coil.ID=ID
   coil.OD=coil[-1].R*2+AWG_dict['size']
   coil.h=h
   coil.r0=center_point
   if simplify is not None:
      coil.simplify(simplify)
   return coil

if __name__=='__main__':
   from AWG import AWG,AWG_square,temperature_adjust,insulate
   wire=temperature_adjust(insulate(AWG_square['12'],style='heavy'), T_degC=22.0)
   coil=square_winder(wire,array([0,0,0.0]),array([0,0,1.0]),0.045,0.131,2.5*.0254*.94,I=20) #,turns_limit=1005)
   print(wire)
   coil.report()
   print('Center B={}'.format(coil.B(coil.r0)))
