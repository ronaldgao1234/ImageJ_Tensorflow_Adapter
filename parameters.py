def import_parameters( file_path ) :
	dict = {}
	for line in open( file_path ) :
		tmp = line.split(' ')
		dict[ tmp[0] ] = tmp[2][:-1]
	return dict
