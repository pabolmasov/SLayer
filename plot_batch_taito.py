from plots import multiplot_saved
from timing import dynspec_maker, specmaker

def plotbatch():
    outlist = ['taito/out_8LRD']
    # , 'out_3HR', 'out_8LR', 'out_8HR', 'out_anc', 'out_NALR', 'out_NAHR']
    for k in outlist:
        print(k+"/...\n")
#        multiplot_saved("titania/"+k+"/runcombine.hdf5_map")
        dynspec_maker(infile=k+'/lcurve0.0', ntimes=10, nbins=150, fmaxout=True)
        dynspec_maker(infile=k+'/lcurve0.7853981633974483', ntimes=10, nbins=150, fmaxout=True)
        dynspec_maker(infile=k+'/lcurve1.5707963267948966', ntimes=10, nbins=150, fmaxout=True)
        specmaker(infile=k+'/lcurve0.0', nbins=200, trange=[0.15,0.2])
        specmaker(infile=k+'/lcurve0.7853981633974483', nbins=200, trange=[0.15,0.2])
        specmaker(infile=k+'/lcurve1.5707963267948966', nbins=200, trange=[0.15,0.2])
        #        dynsplot(infile="titania/"+k+"/pds_mass0.785398163397")
#        pdsplot(infile="titania/"+k+"/pdstots_mass0.785398163397")
#        dynsplot(infile="titania/"+k+"/pds_diss0.785398163397")
#        pdsplot(infile="titania/"+k+"/pdstots_diss0.785398163397")
#        FFplot(prefix="titania/"+k+"/diss_")

    # multireader('out/runcombine.hdf5', derot=True, nframes=1000)
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_8HR/q*.png' -pix_fmt yuv420p -b 4096k titania/out_8HR/q.mp4
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_3HR/q*.png' -pix_fmt yuv420p -b 4096k titania/out_8HR/q.mp4
# ffmpeg -f image2 -r 35 -pattern_type glob -i 'titania/out_3LRinc/q*.png' -pix_fmt yuv420p -b 4096k titania/out_3LRinc/q.mp4
