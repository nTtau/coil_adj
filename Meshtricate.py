# import matplotlib
# matplotlib.use('TkAgg')
from pylab import *
from scipy.integrate import cumtrapz
from scipy.spatial import Delaunay, ConvexHull
from multiprocessing import Manager, Pool,cpu_count
import logging

n_procs=cpu_count()
yes_I_have_wrapped_my_main=False
warning_done=False

logger=logging.getLogger('')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

nodes_per_tri=0.5 # Always applies, no matter what the triangle
nodes_per_tet=4*arccos(23.0/27.0)/(4*pi) # approximation based on a regular tetrahedron, or does this generally apply?
weight_convert_coeff=4.0/(72.0**(1.0/3) * sqrt(3.0))
bulge_ratio=1e-4

def weighted_line(starter,stopper,weightfunc, n_even=100, n_dims=None):
   start=starter.reshape((1,-1))
   stop=stopper.reshape((1,-1))
   if n_dims is None:
      n_dims=start.size
   dist=norm(stop-start)
   even_span=linspace(0.0,1.0,n_even) 
   even_points=start+dot(even_span.reshape((-1,1)),stop-start)
   weights=weightfunc(even_points)
   if n_dims == 2:
      areas=nodes_per_tri/weights # weights= nodes/area, so areas= area/simplex
      nodes_per_meter=sqrt(sqrt(3)/(4*areas)) # approximation based on edge length of a regular triangle
   elif n_dims == 3:
      vols=nodes_per_tet/weights
      nodes_per_meter=(vols*6*sqrt(2))**(-1.0/3) # approximation based on edge length of a regular tetrahedron
   else:
      raise NotImplementedError
   cum_n_nodes=cumtrapz(nodes_per_meter, dx=dist/n_even, initial=0.0) # integrate the nodes/meter along the line
   n_nodes=max(ceil(cum_n_nodes[-1]).astype(int), 1) #  n_nodes doesn't count the start node, but it does count the end node, so we need at least 1.
   cum_n_nodes /= cum_n_nodes[-1] # normalize to [0.0,1.0] 
   selector=interp(linspace(0.0,1.0,n_nodes+1), cum_n_nodes, even_span).reshape((-1,1))
   return start+dot(selector.reshape((-1,1)),stop-start)
   
   



class Meshtricate(object): #Qbb Qc Qx Qz
   def __init__(self, R,weightfunc=None,do_bulge=False,qhull_options=None,n_divide=1,concave=False):
      self.n_orig=R.shape[0]
      self.n_dim=R.shape[1]
      self.do_bulge=do_bulge
      self.bulge_center=mean(R,axis=0)
      #logging.warn('OVERRIDING QHULL OPTIONS')
      self.qhull_options=qhull_options
      self.concave=concave
      if n_procs>1 and n_divide>1 and not yes_I_have_wrapped_my_main:
         print('''
            Warning: You have not wrapped your main routine as follows:
               if __name__=='__main__':
                  (do stuff)
            This is absolutely necessary on Windows for parallelizing the Delaunay tesselation.
            (Alternately, you may have forgotten to set Meshtricate.yes_I_have_wrapped_my_main=True)
            Now dumbing down to single-core computing.
         ''')
         n_divide=1

      if weightfunc is None:
         self.weightfunc=lambda R: ones(R.shape[0])
         # TODO: Weightfunc should really be the desired node density as a function of position;
         # then the boundaries can be refined according to the same weighting.
         # TODO: weightfunc just doesn't pickle well. This causes problems all over, not least with multiprocessing 
      else:
         self.weightfunc=weightfunc
      
      if n_procs==1 or n_divide==1:
         self.Delaunay=Delaunay(self.bulge(R),qhull_options=qhull_options)
#          self.Delaunay.points=R #TODO: Had to disable this for more recent scipy. Figure out right answer (maybe get qhull to do it right)
         #TODO: deal with concave boundary
      else:
         raise NotImplementedError('parallelized tesselation not implemented')
         averageX=R[:,0].mean()
         mesh1=Meshtricate(R[R[0]<averageX, :], weightfunc=weightfunc, qhull_options=qhull_options,n_divide=n_divide/2)
         mesh2=Meshtricate(R[R[0]>=averageX, :], weightfunc=weightfunc, qhull_options=qhull_options,n_divide=n_divide-n_divide/2)
         result=mesh1+mesh2
         self.Delaunay=result.Delaunay
      if concave:
         self.convex=self.Delaunay
         for i,tri in enumerate(self.convex.simplices):
            centr=mean(self.Delaunay.points[tri,:],axis=1).reshape((1,-1))
            
      # calculate volume
      self.volume=0
      self._smidgeon=0
      for i,tri in enumerate(self.Delaunay.simplices):
         self.volume += self.simplex_vol(i)
      self._smidgeon=self.volume/i/1e12 
      if self.Delaunay.simplices.shape[1] == 4: # 3d mesh
         self.nodes_per_simplex=nodes_per_tet
      else: # 2d mesh
         self.nodes_per_simplex=nodes_per_tri
      self.fields={}

   def bulge(self, points):
      if self.do_bulge:
         points_relative=points-self.bulge_center
         points_radius=sqrt(sum(points_relative**2,axis=1)).reshape((-1,1))
         rmax=points_radius.max()
         return self.bulge_center+points_relative*(1.0+bulge_ratio*(1-points_radius/rmax))
      else:
         return points
      
   def get_barycoords(self,locations,simplex_guess=None,outside='nearest'):
      '''
      (simplices, coords) = self.get_barycoords(locations [, simplex_guess=simplex_guess])
         locations: the points you want; shape (n_points, n_dim)
         simplex_guess: which simplices you think may contain those locations; shape (n_points, ); will be overwritten with the found simplices
         outside: what to do when the point isn't inside the mesh.  Options are 'nearest' which finds the nearest simplex, or 'flag' to return simplex=-1 for that point  
         simplices: the enclosing simplex index for each point in locations; shape (n_points, )
         coords: the barycentric coordinates for each point in locations; shape (n_points, n_dim+1)
      '''
      if simplex_guess is None:
         centroid_simplex,dummy=self.get_barycoords(mean(locations,axis=1).reshape((1,-1)), simplex_guess=[0], outside='nearest')
         simplex_guess=ones_like(locations[:,0],dtype=dtype(int_))*centroid_simplex
      else:
         simplex_guess[simplex_guess<0]=0
      coords=empty((locations.shape[0],self.n_dim+1))
      for i in range(locations.shape[0]):
         for n in range(self.Delaunay.simplices.shape[0]): #@UnusedVariable n
            Tinv=self.Delaunay.transform[simplex_guess[i],:self.n_dim,:self.n_dim]
            r=self.Delaunay.transform[simplex_guess[i],self.n_dim,:]
            c=dot(Tinv,locations[i,:]-r) # coordinates with respect to the first (n_dim) vertices (3 out of 4 for a 3D mesh) 
            c=hstack((c,[1.0-sum(c)])) # now we have coordinates with respect to all vertices
            if all(c>=-1e-10): # We found our simplex! (or, we're outside but very very very very close to a vertex of this simplex, so we'll take it)
               coords[i,:]=c
               break
            # if we reach here, our location isn't in this simplex, we need to move to an appropriate neighbor
            which_way, = where(c<0)
            nabors=self.Delaunay.neighbors[simplex_guess[i],which_way]
            if any(nabors>=0):
               simplex_guess[i]=max(nabors)
               continue
            # Now we have to deal with the case where the point is outside the mesh.
            if outside == 'flag':
               simplex_guess[i] = -1
               break
            elif outside == 'nearest':
               #Assume a convex hull, so best simplex will include the nearest 
               raise NotImplementedError # TODO: fix, this should be the default
      return (simplex_guess.copy(), coords)
      
   def simplex_vol(self,simplex):
      tri=self.Delaunay.simplices[simplex,:]
      points=self.Delaunay.points[tri,:]
      points -= points[0,:]
      vol=cross(points[1,:], points[2,:])/2.0
      if len(tri) == 4: # 3D mesh
         vol=norm(dot(vol,points[3,...]))/3.0
         return abs(vol)
      else: # 2D mesh
         return norm(vol)
      
   def condition(self,n_iter=5):
      logger.info('Conditioning, {} passes'.format(n_iter))
      # Weighted Lloyd algorithm for Centroidal Vornoi Tesselation
      for ni in range(n_iter): # @UnusedVariable ni
         #logger.debug('collecting simplices for each vertex and calculating simplex parameters')
         #Vornoi diagram.  First, collect all simplices for a vertex
         vertex2simplices=[[] for nv in range(self.Delaunay.points.shape[0])]
         simplex_centers=zeros( (self.Delaunay.simplices.shape[0], self.n_dim) )
         simplex_weights=zeros(self.Delaunay.simplices.shape[0])
         for tri_idx,simplex in enumerate(self.Delaunay.simplices):
            for vert in simplex:
               vertex2simplices[vert].append(tri_idx)
            simplex_centers[tri_idx,:]=sum(self.Delaunay.points[simplex,:],axis=0)/len(simplex)
            # TODO: since weightfunc doesn't pickle well, should we just require weights to be provided at the nodes?
            weight=self.weightfunc(simplex_centers[tri_idx,:].reshape((1,-1)))[0]
            vol=self.simplex_vol(tri_idx)
            simplex_weights[tri_idx]=vol*weight
            #logger.debug('simplex {} vol: {}, weight: {}'.format(tri_idx, vol, weight))
         # Now calculate the weighted Vornoi centroid for this vert
         #logger.debug('calculating vornoi centroids')
         new_R=self.Delaunay.points.copy()
         weighted_centers=simplex_centers*simplex_weights.reshape((-1,1))
         #logger.debug('shifting points')
         for nv,simplices in enumerate(vertex2simplices):
            if nv<self.n_orig: #boundary node, don't budge
               continue
            weightsum=sum(simplex_weights[simplices])
            if weightsum>0:
               new_R[nv,:]=sum(weighted_centers[simplices,:],axis=0)/weightsum
            #logger.debug('here: nv={}, new_R={}'.format(nv, new_R[nv,:]))
         logger.debug('Re-tesselating')
         self.Delaunay=Delaunay(self.bulge(new_R),qhull_options=self.qhull_options)
#          self.Delaunay.points=new_R #TODO: Had to disable this for more recent scipy. Figure out right answer (maybe get qhull to do it right)
         logger.debug('Rebuild complete')
      
   def interpolate(self,values,locations,outside='fill',simplex_guess=None):
      # TODO: allow values to be array [n_points x n_components], e.g. for interpolating B or E
      #       cf. CurrentVolume.Delaunay.interpolator
      simplices,coords=self.get_barycoords(locations,simplex_guess=simplex_guess,outside='flag')
      results=empty((locations.shape[0],),dtype=values.dtype)
      for i,simplex in enumerate(simplices):
         if (simplex == -1):
            try:
               if outside.lower() == 'fill': 
                  dist=sum((self.Delaunay.points - locations[i,:])**2)
                  nearest=dist.argmin()
                  results[i]=values[nearest]
            except:
               results[i]=outside
         else:
            results[i]=dot(coords[i,:],values[self.Delaunay.simplices[simplex,:]])
      return results          
         
   def refine(self, min_points=0, condition=None):
      logger.info('refining')
      while True:
         if condition is None:
            self.condition(rint((self.Delaunay.points.shape[0])**(1.0/self.n_dim)).astype(int))
         elif condition:
            self.condition(condition)
         quality=zeros((self.Delaunay.simplices.shape[0]))
         target_Npts=0.0
         current_Npts=0.0 # counting them, rather than taking the shape of the points array, allows proper accounting of "half nodes" at the boundary 
         for i,tri in enumerate(self.Delaunay.simplices):
            weights=self.weightfunc(self.Delaunay.points[tri]).reshape((-1,1)) # nodes/vol or nodes/area
            density=mean(weights)
            vol=self.simplex_vol(i)
            this_demand=density*vol
            #logger.debug('density={}, vol={}, this_demand: {}'.format(density, vol, this_demand))
            target_Npts += this_demand
            current_Npts += self.nodes_per_simplex
            quality[i]=self.nodes_per_simplex-this_demand
         target_Npts=max(min_points,target_Npts*0.999)
         logger.info('target_Npts={}, currently have {}'.format(target_Npts, current_Npts))
         N_refine=rint(max(target_Npts - current_Npts, 0)).astype(int)
         logger.debug('# points to refine: {}'.format(N_refine))
         if N_refine>0:
            points_to_add=[]
            is_refined=zeros_like(quality, dtype=bool_) # not refined == tacky
            for tri_idx in quality.argsort()[:N_refine]:
               tri=self.Delaunay.simplices[tri_idx,:]
               weights=self.weightfunc(self.Delaunay.points[tri,:]).reshape((-1,1))
               centroid=sum(self.Delaunay.points[tri,:]*weights,axis=0)/float(sum(weights))
               neighbors=self.Delaunay.neighbors[tri_idx]
               neighbor_refined=False
               for neighbor in neighbors:
                  if neighbor >=0:
                     neighbor_refined=is_refined[neighbor]
               if not neighbor_refined:
                  points_to_add.append(list(centroid))
                  is_refined[tri_idx]=True
            #self.Delaunay.add_points(points_to_add) #doesn't seem to work
            new_R=vstack((self.Delaunay.points,array(points_to_add)))
            self.Delaunay=Delaunay(self.bulge(new_R),qhull_options=self.qhull_options)
#             self.Delaunay.points=new_R  #TODO: Had to disable this for more recent scipy. Figure out right answer (maybe get qhull to do it right)
         else:
            break

   def plot(self, **kwargs):
      triplot(self.Delaunay.points[:,0],self.Delaunay.points[:,1],self.Delaunay.simplices, **kwargs)
   
   def render(self, **kwargs):
      from mayavi import mlab as Mlab
      if 'name' not in kwargs:
         try:
            kwargs['name']=self.name
         except:
            kwargs['name']="Meshtricate Object"
      if 'color' not in kwargs:
         try:
            kwargs['color']=self.color
         except:
            kwargs['color']=(0.0,0.0,1.0)
      if 'opacity' not in kwargs:
         try:
            kwargs['opacity']=self.opacity
         except:
            kwargs['opacity']=1.0
      if 'scale_mode' not in kwargs:
         try:
            kwargs['scale_mode']=self.scale_mode
         except:
            kwargs['scale_mode']='vector'
      if 'mode' not in kwargs:
         try:
            kwargs['mode']=self.render_mode
         except:
            kwargs['mode']='cylinder'
      links=[[] for point in self.Delaunay.points[:,0]] #@UnusedVariable point
      #logging.warn('ONLY RENDERING INTERIOR TET EDGES')
      for simplex in self.Delaunay.simplices:
         sordid=sort(simplex)
         for i,node1 in enumerate(sordid):
            for node2 in sordid[i+1:]:
               #if node2>=self.n_orig:
               links[node1].append(node2)
      v1=[]
      v2=[]
      for node1,linklist in enumerate(links):
         for node2 in linklist:
            v1.append(node1)
            v2.append(node2)
      points=self.Delaunay.points[v1,:]
      lengths=self.Delaunay.points[v2,:]-points
      glyphs=Mlab.points3d(self.Delaunay.points[:,0], self.Delaunay.points[:,1], self.Delaunay.points[:,2])
      for n,pt in enumerate(self.Delaunay.points):
         Mlab.text(pt[0],pt[1],'{}'.format(n),z=pt[2], color=(0.0,0.0,0.0),width=0.05)
      glyphs.glyph.glyph.scale_factor = 0.05
      vectors=Mlab.quiver3d(points[:,0], points[:,1], points[:,2],
                            lengths[:,0], lengths[:,1], lengths[:,2],
                            **kwargs)
      vectors.glyph.glyph.clamping=False
      vectors.glyph.glyph.scale_factor=1.0
      vectors.glyph.glyph_source.glyph_source.radius = 0.005
      #Mlab.text(0.0,0.0,'Origin', z=0.0, color=(0.0,0.0,0.0),width=0.25)
      return vectors 

   def gradient_at_nodes(self,values):
      result=empty((values.size,self.n_dim), dtype=values.dtype)
      indices,indptr=self.Delaunay.vertex_neighbor_vertices
      for vert in range(self.Delaunay.points.shape[0]):
         neighbors=concatenate(( indptr[indices[vert]:indices[vert+1]], [vert] ))
         A=hstack(( self.Delaunay.points[neighbors,:], ones((neighbors.size,1)) ))
         b=values[neighbors]
         gradP=lstsq(A,b)[0] 
         result[vert,:]=gradP[:-1]
      return result
   
   def boundary_simplices(self,thick=False):
      if thick:
         bdy_simps,dummy=where(self.Delaunay.simplices < self.n_orig)
      else:
         bdy_simps,dummy=where(self.Delaunay.neighbors < 0)
      return unique(bdy_simps)   
   
   def boundary_facets(self):
      simplices=self.boundary_simplices()
      
   
   def is_inside(self,locations):
      simplices,dummy=self.get_barycoords(locations, outside='flag')
      return (simplices >= 0)
         
   
   def __add__(self,other):
      raise NotImplementedError('adding triangulations') #TODO: implement. Don't forget self.fields
      #TODO: can we leverage self.boundary_simplices or self.boundary_facets?
      # Step 1: for each mesh, collect the points comprising the convex hull (hull_indicess),
      #                                the points in any simplex that touches the convex hull (shell_indices), and
      #                                the indices for any simplex that does not touch the hull (remaining_tets). 
      hull_indices=[]
      shell_indices=[]
      remaining_tets=[]
      for mesh in [self,other]:
         
         # TODO: compare these methods
         hull=mesh.Delaunay.convex_hull
         #hull=ConvexHull(mesh.Delaunay.points).simplices
         
         hull_indices.append(unique(hull))
         this_shell_indices=[]
         this_remaining_tets=[]
         for tet in range(mesh.Delaunay.simplices.shape[0]):
            found=False
            for vert in mesh.Delaunay.simplices[tet,:]:
               if vert in hull_indices[-1]:
                  found=True
                  break
            if found:
               this_shell_indices.append(mesh.Delaunay.simplices[tet,:])
            else:
               this_remaining_tets.append(mesh.Delaunay.simplices[tet,:])
         shell_indices.append(unique(hstack(this_shell_indices)))
         remaining_tets.append(vstack(this_remaining_tets))
      # Step 2: Tesselate the combined shells (shell = hull + one layer of simplices) of the two meshes.
      #         This will generate some tets that span the two addends (good),
      #                            some that are duplicates of what exists in a single shell (ok), and
      #                            some that are inside the shell of one mesh (bad).  
      #TODO: clean out repeated points in glue (from eg. points on a reflection plane)
      gluepoints=vstack(( self.Delaunay.points[shell_indices[0],:], other.Delaunay.points[shell_indices[1],:] ))
      #TODO: implement "bulge"
      glue=Delaunay(gluepoints, qhull_options=self.qhull_options)
      # Step 3: Find the simplices of the "glue" that aren't completely interior to one of the two addends' shells  
      glue_keepers=[]
      for tet in range(glue.Delaunay.simplices.shape[0]):
         if any(glue.simplices[tet,:] < shell_indices[0].shape[0]) and any(glue.simplices[tet,:] >= shell_indices[0].shape[0]):
            # the tet spans the two meshes, so keep it
            glue_keepers.append(tet)
            continue
         for vertex in glue.Delaunay.simplices[tet,:]:
            if vertex<shell_indices[0].shape[0]:
               # the point is in the first mesh
               if shell_indices[0][vertex] in hull_indices[0]:
                  # the tet touches the first hull, so keep it
                  glue_keepers.append(tet)
                  break
            else:
               # the point is in the second mesh
               if shell_indices[1][vertex-shell_indices[0].shape[0]] in hull_indices[1]:
                  # the tet touches the second hull, so keep it
                  glue_keepers.append(tet)
                  break
         result_points=vstack(( self.Delaunay.points, other.Delaunay.points ))
         result_simplices=this_remaining_tets[0]
         for tet in this_remaining_tets[1]:
            result_simplices.append(tet+self.Delaunay.points.shape[0])
         for tet in glue_keepers:
            result_tet=zeros_like(tet)
            # convert to indices in the "self" mesh region of the result
            mask= (tet < shell_indices[0].shape[0])
            result_tet[mask] = shell_indices[0][tet[mask]] 
            # convert to indices in the "other" mesh region of the result
            mask= (tet >= shell_indices[0].shape[0])
            other_shell_indices=tet[mask]-shell_indices[0].shape[0]
            other_mesh_indices=shell_indices[1][other_shell_indices]
            result_tet[mask] = other_mesh_indices+self.Delaunay.points.shape[0]
      
   

class TwoDin3D(Meshtricate):
   def __init__(self, R,weightfunc=None,do_bulge=False,qhull_options=None,n_divide=1):
      if n_procs>1 and n_divide>1 and not yes_I_have_wrapped_my_main:
         print('''
            Warning: You have not wrapped your main routine as follows:
               if __name__=='__main__':
                  (do stuff)
            This is absolutely necessary on Windows for parallelizing the Delaunay tesselation.
            (Alternately, you may have forgotten to set Meshtricate.yes_I_have_wrapped_my_main=True)
            Now dumbing down to single-core computing.
         ''')
         n_divide=1
      self.do_bulge=do_bulge
      self.qhull_options=qhull_options
      self.R0=R[0,:] # remember the 3D origin so we can translate 2D coordinates into 3D
      self.vec1=R[1,:]-R[0,:] # the first 2 points provided are designated the new x-axis
      self.vec1 /= norm(self.vec1) # vec1 is now the x-direction unit vector
      coord1=self.vec1*dot(R,self.vec1.T).reshape(-1,1) # the x-component of the point locations
      tempvec2=R-coord1 # tempvec2 is the projection of the points onto the new y-z plane
      tempvec2 -= tempvec2[0,:] # the first point defines the yz-origin 
      absvec2=sqrt(sum(tempvec2**2,axis=1)) # distance to all the points in the y-z plane
      goodchoice=absvec2.argmax() # the furthest point from the origin defines a good y-axis
      self.vec2=tempvec2[goodchoice,:]
      self.vec2 /= norm(self.vec2) # vec2 is now the y-direction unit vector
      R_flat=hstack( (dot(R-self.R0,self.vec1.T).reshape(-1,1), dot(R-self.R0,self.vec2.T).reshape(-1,1)) ) # 2D coordinates
      
      self.n_orig=R.shape[0]
      self.n_dim=2
      self.nodes_per_simplex=nodes_per_tri
      if weightfunc is None:
         self.weightfunc_3D=lambda R: ones(R.shape[0])
      else:
         self.weightfunc_3D=weightfunc
      
      self.bulge_center=mean(R_flat,axis=0)
      if n_procs==1 or n_divide==1:
         self.Delaunay=Delaunay(self.bulge(R_flat),qhull_options=qhull_options)
#          self.Delaunay.points=R_flat #TODO: Had to disable this for more recent scipy. Figure out right answer (maybe get qhull to do it right)
      else:
         raise NotImplementedError #TODO: implement
         averageX=R_flat[:,0].mean()
         mesh1=Meshtricate(R_flat[R_flat[:,0]<averageX, :], weightfunc=weightfunc, qhull_options=qhull_options,n_divide=n_divide/2)
         mesh2=Meshtricate(R_flat[R_flat[:,0]>=averageX, :], weightfunc=weightfunc, qhull_options=qhull_options,n_divide=n_divide-n_divide/2)
         result=mesh1+mesh2
         self.Delaunay=result.Delaunay

      volume=0
      self._smidgeon=0
      for i,tri in enumerate(self.Delaunay.simplices):
         volume += self.simplex_vol(i)
      self._smidgeon=volume/len(self.Delaunay.simplices)/1e12
      
   def translate_2D(self,locations):
      return hstack( (dot(locations-self.R0,self.vec1.T).reshape(-1,1), dot(locations-self.R0,self.vec2.T).reshape(-1,1)) )

   def translate_3D(self,locations):
      return self.R0 + self.vec1*locations[:,0:1] + self.vec2*locations[:,1:2]
   
   def weightfunc(self,R, *args, **kwargs):
      R3D=(R[:,0].reshape((-1,1))*self.vec1.reshape((1,-1)) +
           R[:,1].reshape((-1,1))*self.vec2.reshape((1,-1)) )
      return weight_convert_coeff * self.weightfunc_3D(R3D, *args, **kwargs)**(2.0/3.0)
   
   def Points3D(self):
      return dot(self.Delaunay.points, vstack( (self.vec1, self.vec2) ))+self.R0 
   
   def render(self, **kwargs):
      from mayavi import mlab as Mlab
      if 'name' not in kwargs:
         try:
            kwargs['name']=self.name
         except:
            kwargs['name']="TwoDin3D Surface"
      if 'color' not in kwargs:
         try:
            kwargs['color']=self.color
         except:
            kwargs['color']=(0.0,0.0,1.0)
      if 'opacity' not in kwargs:
         try:
            kwargs['opacity']=self.opacity
         except:
            kwargs['opacity']=0.5
      points=self.Points3D()
      return Mlab.triangular_mesh(points[:,0], points[:,1], points[:,2], self.Delaunay.simplices, **kwargs)

         
if __name__=='__main__':
   import sys
   lumberjack=logging.StreamHandler(sys.stdout)
   logger.addHandler(lumberjack)
   logger.setLevel(logging.INFO)
   lumberjack.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
   logger.info('Welcome to Meshtricate!')
   width=1.0
   height=width*sqrt(0.75)
   weight_stddev=width/7
   peak_density=1000/(0.5*width*height)
   min_density=peak_density/50
   def weight_demo(points):
      dx=points[:,0]-width/2
      dy=points[:,1]-height/3
      return min_density+(peak_density-min_density)*exp(- (dx**2 + dy**2)/weight_stddev**2 )
   ptA=array([0.0,0.0])
   ptB=array([width/2,height])
   ptC=array([width,0.0])
   lineAB=weighted_line(ptA, ptB, weight_demo)[1:,:]
   lineBC=weighted_line(ptB, ptC, weight_demo)[1:,:]
   lineCA=weighted_line(ptC, ptA, weight_demo)[1:,:]
   edges=vstack(( lineAB, lineBC, lineCA ))
   logger.info('Generating initial triangulization')
   mesh=Meshtricate(vstack(edges), weightfunc=weight_demo, do_bulge=True)
   logger.info('Refining / adding points')
   mesh.refine()
   figure(figsize=(15.0,10.00))
   logger.info('Plotting')
   subplot(121)
   weights=mesh.weightfunc(mesh.Delaunay.points)
   tripcolor(mesh.Delaunay.points[:,0],mesh.Delaunay.points[:,1],mesh.Delaunay.simplices,weights)
   maxwt=weights.max()
   hold(True)
   mesh.plot(color='black')
   axis('equal')
   colorbar()
   title('Weights')
   subplot(122)
   densities=zeros((mesh.Delaunay.simplices.shape[0],))
   for idx in range(mesh.Delaunay.simplices.shape[0]):
      densities[idx]=min(mesh.nodes_per_simplex/mesh.simplex_vol(idx),maxwt)
   tripcolor(mesh.Delaunay.points[:,0],mesh.Delaunay.points[:,1],mesh.Delaunay.simplices,densities)
   hold(True)
   mesh.plot(color='black')
   axis('equal')
   colorbar()
   title('Densities')
   show()
   import cPickle
   spickle=cPickle.dumps(mesh)
   d3=dot(mesh.Delaunay.points,array([[1,0,0],[0,1,0]]))
   mesh2=TwoDin3D(d3, weightfunc=weight_demo, do_bulge=True)
   spickle=cPickle.dumps(mesh2)
