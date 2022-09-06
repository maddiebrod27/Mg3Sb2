def coord_cart(vec, lat_vecs):
	"""
	converts coordinates or vectors from the lattice to Cartesian basis.
		args:
		vec: array_like shape (3,) 
		lat_vecs (array_like): shape (3, 3) rows contain the lattice vectors in terms of cartesian space
	"""
	aij = lat_vecs.T
	return np.matmul(aij, vec.T)

def find_neighbors(atom_label, atom_pos_dict, lat_vec, tol = 1E-4, cutoff = 16):
	"""
	Finds all of the neighbors for a specific atomic site in a crystalline materials up to 
	a given cutoff number of neighbors (cutoff).
	
	args:
		atom_label (str): label of central atom
		atom_pos (dict): dictionary containing all atom labels and corresponding positions (numpy array or list)
							the positions are given in the lattice basis
		lat_vec (array_like): contains lattice vectors in terms of cartesian vectors
		tol (float): tolerance for determining if neighbor is the same atom
		cutoff (int): number of neighbors to find
	returns:
		dict: dictionary containing the label of the neighbors and their positions (in the lattice basis)
	"""
	


	min_distance = 1E6
	
	central_atom_pos = atom_pos_dict[atom_label]
	central_atom_pos_cart = coord_cart(central_atom_pos, lat_vec)
	
	periodic_data = {}
	all_atom_types = list(atom_pos_dict.keys())


	all_atom_pos = list(atom_pos_dict.values())
	aij = lat_vec.T

	for atom_type, atom_pos in zip(all_atom_types, all_atom_pos):
		atom_pos_cart = coord_cart(atom_pos, lat_vec)
		atom_label_index = 1
		
		#search through periodic images for neighbors
		for i in range(-1, 2):
			for j in range(-1, 2):
				for k in range(-1, 2):
					
					# Calculate coordinate of periodic image
					#new_pos = atom_pos + (i*(cell_dim+2))*lat_vec[0]+ (j*(cell_dim+2))*lat_vec[1] + (k*(cell_dim+2))*lat_vec[2]
					new_pos = atom_pos_cart + i*lat_vec[0]+ j*lat_vec[1] + k*lat_vec[2]
					if (np.linalg.norm(central_atom_pos_cart-new_pos) < tol):	# Check to see if the distance is 0 (same atom)
						continue
					
					# Calculate distance to central atom
					distance = np.linalg.norm(central_atom_pos_cart-new_pos)
					if distance < min_distance:
						min_distance = distance

					# Record data
					periodic_data[atom_type + "_" + str(atom_label_index)] = {	"Position": np.dot(np.linalg.inv(aij), new_pos),
																				"Position_Cart": new_pos,
																				"Distance": round(distance,3)	}
					
					# Update label of atom
					atom_label_index += 1
	neighbor_distances = []
	for k, v in sorted(periodic_data.items(), key=lambda item: item[1]["Distance"]):
		distance = v["Distance"]
		
		if len(neighbor_distances) == 0:
			neighbor_distances.append(distance)
		else:
			if all( (np.abs(distance - ndist) > tol) for ndist in neighbor_distances):
				neighbor_distances.append(distance)

	NN_distances = neighbor_distances[0:16]

	
	NN_positions = {}

	d = 0
	marker = 1
	counter = 0
	for k, v in sorted(periodic_data.items(), key=lambda item: item[1]["Distance"]):
	
		
		d = v["Distance"]
		if d <= NN_distances[-1]:
			#if d <= 6:
			NN_positions[k.split("_")[0] + "_" + k.split("_")[1] + "_" + str(marker)] = v["Position"]
			marker += 1


		counter += 1
	return NN_positions




def calc_hamil(neighbor_vecs, lat_vecs, tbparams, on_sites, k):
	"""
	Calculates the tight-binding Hamiltonian for a toy Mg3Sb2 model at a given k-point, given the basis,
	Mg(1)-s, Mg(2)_1-s, Mg(2)_2-s, Sb_1-s, Sb_1-px, Sb_1-py, Sb_1-pz, Sb_2-s, Sb_2-px, Sb_2-py, Sb_2-pz.

	args:
		neighbors_vecs (dict): nested dictionary containing the lattice and Cartesian neighbor vectors for specified
								neighbors pairs
		
		lat_vecs (array_like): (3,3) array containing the lattice vectors with respect to a Cartesian basis
		
		tbparams (dict): dictionary containing the tight-binding interaction parameters for a given orbital pair
							of a given bonding type (sigma or pi)

		on_sites (array_like): list or array of shape (11,) that contains the on-site energies in the order
								of the basis

		k (array_like): shape array or list of shape (3,) that gives the k-point in the reciprocal lattice
							basis (multiplied by 2*pi)
	"""
	dim = 11 #dimension of atomic orbital basis
	hamil = np.zeros((dim, dim), dtype = complex)

	for i in range(dim):
		hamil[i][i] = on_sites[i]


	
	for n in neighbor_vecs["Lattice"]["Sb-1_Sb-2"][0:3]:
		
		#convert vector between neighbors from lattice to Cartesian basis
		n_c = coord_cart(n, lat_vecs)
		
		#find components of cosine vector between neighbors
		l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
		m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
		nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
	

		#use Bloch sums to calculate Hamiltonian element
		hamil[4][8] += cmath.exp( 1j * np.dot(k, n) ) * (
			l**2 * tbparams["Sb-1_Sb-2_p_p_sig"] + (1 - l**2) * tbparams["Sb-1_Sb-2_p_p_pi"])
		
		hamil[4][9] += cmath.exp( 1j * np.dot(k, n) ) * (
			l*m * tbparams["Sb-1_Sb-2_p_p_sig"] - l*m * tbparams["Sb-1_Sb-2_p_p_pi"])
		
		hamil[4][10] += cmath.exp( 1j * np.dot(k, n) ) * (
			l*nn * tbparams["Sb-1_Sb-2_p_p_sig"] - l*nn * tbparams["Sb-1_Sb-2_p_p_pi"])

		
		hamil[5][9] += cmath.exp( 1j * np.dot(k, n) ) * (
			m**2 * tbparams["Sb-1_Sb-2_p_p_sig"] + (1 - m**2) * tbparams["Sb-1_Sb-2_p_p_pi"])
		
		hamil[5][8] += cmath.exp( 1j * np.dot(k, n) ) * (
			l*m * tbparams["Sb-1_Sb-2_p_p_sig"] - l*m * tbparams["Sb-1_Sb-2_p_p_pi"])
		
		hamil[5][10] += cmath.exp( 1j * np.dot(k, n) ) * (
			m*nn * tbparams["Sb-1_Sb-2_p_p_sig"] - m*nn * tbparams["Sb-1_Sb-2_p_p_pi"])
	
		
		hamil[6][10] += cmath.exp( 1j * np.dot(k, n) ) * (
			nn**2 * tbparams["Sb-1_Sb-2_p_p_sig"] + (1 - nn**2) * tbparams["Sb-1_Sb-2_p_p_pi"])
		
		hamil[6][8] += cmath.exp( 1j * np.dot(k, n) ) * (
			nn*l * tbparams["Sb-1_Sb-2_p_p_sig"] - nn*l * tbparams["Sb-1_Sb-2_p_p_pi"])
		
		hamil[6][9] += cmath.exp( 1j * np.dot(k, n) ) * (
			m*nn * tbparams["Sb-1_Sb-2_p_p_sig"] - m*nn * tbparams["Sb-1_Sb-2_p_p_pi"])

	for n in neighbor_vecs["Lattice"]["Mg-o_Sb-1"][0:3]:
		#print(n)
		n_c = coord_cart(n, lat_vecs)
		l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
		m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
		nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
		
		hamil[0][3] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-o_Sb-1_s_s"]
		
		hamil[0][4] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-o_Sb-1_s_p"] * l
		hamil[0][5] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-o_Sb-1_s_p"] * m
		hamil[0][6] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-o_Sb-1_s_p"] * nn

	for n in neighbor_vecs["Lattice"]["Mg-o_Sb-2"][0:3]:
		#print(n)
		n_c = coord_cart(n, lat_vecs)
		l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
		m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
		nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
		hamil[0][7] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-o_Sb-1_s_s"]
		
		hamil[0][8] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-o_Sb-1_s_p"] * l
		hamil[0][9] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-o_Sb-1_s_p"] * m
		hamil[0][10] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-o_Sb-1_s_p"] * nn
	



	n  =  neighbor_vecs["Lattice"]["Mg-t1_Sb-1"][0]
	n_c = coord_cart(n, lat_vecs)
	l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
	m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
	nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
	hamil[1][3] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-1_s_s"]
	
	hamil[1][4] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-t1_Sb-1_s_p"] * l
	hamil[1][5] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-t1_Sb-1_s_p"] * m
	hamil[1][6] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-t1_Sb-1_s_p"] * nn
	
	for n in neighbor_vecs["Lattice"]["Mg-t1_Sb-2"][0:3]:
		#print(n)
		n_c = coord_cart(n, lat_vecs)
		l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
		m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
		nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
		hamil[1][7] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-2_s_s"]
		
		hamil[1][8] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-t1_Sb-2_s_p"] * l
		hamil[1][9] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-t1_Sb-2_s_p"] * m
		hamil[1][10] += cmath.exp( 1j * np.dot(k, n) )* tbparams["Mg-t1_Sb-2_s_p"] * nn
	


	for n in neighbor_vecs["Lattice"]["Mg-t2_Sb-1"][0:3]:
		n_c = coord_cart(n, lat_vecs)
		l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
		m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
		nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
		hamil[2][3] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-2_s_s"]
		
		hamil[2][4] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-2_s_p"] * l
		hamil[2][5] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-2_s_p"] * m
		hamil[2][6] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-2_s_p"] * nn

	
	n = neighbor_vecs["Lattice"]["Mg-t2_Sb-2"][0]
	n_c = coord_cart(n, lat_vecs)
	l = np.dot(n_c, np.array([1, 0, 0])) / np.linalg.norm(n_c)
	m = np.dot(n_c, np.array([0, 1, 0])) / np.linalg.norm(n_c)
	nn = np.dot(n_c, np.array([0, 0, 1])) / np.linalg.norm(n_c)
	hamil[2][7] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-1_s_s"]
	
	hamil[2][8] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-1_s_p"] * l
	hamil[2][9] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-1_s_p"] * m
	hamil[2][10] += cmath.exp( 1j * np.dot(k, n) ) * tbparams["Mg-t1_Sb-1_s_p"] * nn


	#fill in lower diagonal of Hamiltonian matrix such that matrix is Hermitian
	for i in range(dim):
		for j in range(dim):
			if i > j:
				hamil[i][j] = np.conjugate(hamil[j][i])
	return hamil


#define basis of primitive lattice
z1 = 0.37
z2 = 0.23

#o = octahedral (coordination)
#t = tetrahedral (coordination)
atom_pos_dict = {
	
	"Mg-o": np.array([0, 0, 0]),
	"Mg-t1": np.array([2/3, 1/3, z1]),
	"Mg-t2": np.array([1/3, 2/3, 1-z1]),
	"Sb-1": np.array([2/3, 1/3, 1-z2]),
	"Sb-2": np.array([1/3, 2/3, z2])
}




#define lattice vectors in terms of Cartesian space
lat_vecs = np.array([
	[a/2, -a*np.sqrt(3)/2, 0],
	[a/2, a*np.sqrt(3)/2, 0],
	[0, 0, c]
])


#creates a nested dictionary that stroes the neighbor vectors in the Cartesian and lattice basis
cart = {}
lat = {}
neighbor_vecs = {"Cartesian": cart,
				"Lattice": lat
}
	
	

for item in list(atom_pos_dict.keys()):
	neighbors = find_neighbors(item, atom_pos_dict, lat_vecs) #in lattice coordinates

	#convert position of atom to Cartesian
	atom_pos_cart = coord_cart(atom_pos_dict[item], lat_vecs)
	for elem, pos in zip(list(neighbors.keys()), neighbors.values()):
		atom_name = elem.split("_")[0]
		pos_cart = coord_cart(pos, lat_vecs)
		vec = pos - atom_pos_dict[item]
		vec_cart = pos_cart - atom_pos_cart
		
		#create key label for the neighbor pair
		pair_name = item + "_" + atom_name
		
		if pair_name in list(neighbor_vecs["Lattice"].keys()):
			neighbor_vecs["Cartesian"][pair_name] = np.vstack([neighbor_vecs["Cartesian"][pair_name], vec_cart])
			neighbor_vecs["Lattice"][pair_name] = np.vstack([neighbor_vecs["Lattice"][pair_name], vec])
		else:
			neighbor_vecs["Cartesian"][pair_name] = vec


#define dictionary that contains tight-binding interaction parameters
tbparams = {}


#list of possible interactions
interactions = [
					"Mg-o_Sb-1_s_p",
					"Mg-t1_Sb-1_s_p",
					"Mg-t1_Sb-2_s_p",
					"Mg-o_Sb-1_s_s",
					"Mg-t1_Sb-2_s_s",
					"Mg-t1_Sb-1_s_s",
					"Sb-1_Sb-2_p_p_pi",
					"Sb-1_Sb-2_p_p_sig"
					]

#initialize each interaction to zero
for interaction in interactions:
	tbparams[interaction] = 0

	
#assign interaction parameters a value (simply comment out to reset an interaction to zero)


#Mg-s/Sb-p interactions
tbparams["Mg-o_Sb-1_s_p"] =  1.4
tbparams["Mg-t1_Sb-1_s_p"] = 1.8 #longer bond length
tbparams["Mg-t1_Sb-2_s_p"] =  2. #shorter bond length

#Mg-s/Sb-s interactions
tbparams["Mg-o_Sb-1_s_s"] =  -0.8
tbparams["Mg-t1_Sb-1_s_s"] = -1.0 #longer bond length
tbparams["Mg-t1_Sb-2_s_s"] =  -1.3 #shorter bond lengthå

#Sb-p/Sb-p interactions
tbparams["Sb-1_Sb-2_p_p_sig"] =  1.0
tbparams["Sb-1_Sb-2_p_p_pi"] =  -0.1
