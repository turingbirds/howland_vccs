#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Execute a child program in a new process"""
import os
import subprocess



def exec_binary(cmd_line_list, cwd=None, stdout=None, stderr=None, verbose=False):
	"""Invoke an executable file.

	Example
	-------

	Easy:
	>>> exec_binary('ls', verbose=True)

	Pass parameters:
	>>> cmd_line_list = [os.path.join('mypathtobin', 'mybin.exe'), '-conf', myparameter]
	>>> if verbose:
	>>> 	cmd_line_list += ['-v']
	>>> exec_binary(cmd_line_list, verbose=verbose)

	Capturing stdout:
	>>> import tempfile
	>>> [out_fd, out_fname] = tempfile.mkstemp()
	>>> try:
	>>> 	exec_binary(cmd_line_list, cwd=bindir, stderr=out_fd, verbose=False)
	>>> finally:
	>>> 	os.fsync(out_fd)
	>>> 	os.lseek(out_fd, 0, os.SEEK_SET)
	>>> 	out = os.read(out_fd, 100)	# captures stdout
	>>> 	os.close(out_fd)
	>>> 	cleanup_files([out_fname])


	Parameters
	----------
	cmd_line_list : list
		Passed as first argument to subprocess.Popen.
	cwd : string
		If not ``None``, change current directory to ``cwd``. Old cwd will be
		restored after exit.
	stdout : None or open, writeable file stream
		If provided, redirect stdout to the given file stream.
	stderr : None or open, writeable file stream
		If provided, redirect stderr to the given file stream.
	verbose : boolean
		If False, redirect tool stdout and stderr to /dev/null, unless stdout
		parameter is given.

	"""

	if verbose:
		print '* Invoking external tool...'
		print '  Command line is: ' + ' '.join(cmd_line_list)

	# set cwd
	if cwd != None:
		old_cwd = os.getcwd()
		os.chdir(cwd)

	try:
		if not verbose:
			devnull = open('/dev/null','w')
			if stdout == None:
				stdout = devnull
			if stderr == None:
				stderr = devnull

		try:
			# invoke the tool
			proc = subprocess.Popen(cmd_line_list, stdout=stdout, stderr=stderr, close_fds=True)
			proc.wait()
		finally:
			if not verbose:
				devnull.close()
			# if not stderr is None:
				# proc.stderr.close()
			# if not stdout is None:
				# proc.stdout.close()

		if proc.returncode > 0:
			raise Exception('Error executing external tool! Return code ' + str(proc.returncode))

	finally:
		# restore old cwd
		if cwd != None:
			os.chdir(old_cwd)



def cleanup_files(fname_list, verbose=False):
	for fname in fname_list:
		if verbose:
			print '* Removing file \'' + fname + '\''

		try:
			os.remove(fname)
		except:
			print '/!\ Warning: could not remove file ' + fname

