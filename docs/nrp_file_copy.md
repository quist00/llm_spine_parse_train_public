trying to upload large zip failed
trying to scp failed
trying to uload through gui succeeds in chunks


count files
ls -1 | wc -l

generate full files list on local
    ls > ~/Desktop/full_file_list.txt

save missing files on remote
    comm -23 <(sort full_file_list.txt) <(ls | sort) > missing_files.txt

copy missing to directory for upload on local
    comm -23 <(sort full_file_list.txt) <(ls | sort) > missing_files.txt

Upload through gui