#!/bin/python3

from struct   import unpack
from typing   import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta

import numpy as np

from .utils import (_decompress, _skip_n_lines)

class CalpuffOutput:
    def __init__(self):

        # File metadata
        self.filepath: str = ""                     
        self.file_type: str = ""                    # 'CONC', 'DFLX', 'WFLX', 'VISB'. 'RHO', 'T2D', 'FOG'
        self.model_version: str = ""                # > 7.2.1
        self.title: str = ""

        #Internal vars: (used to handle data)
        self.NCOM: int = 0                          # Number of commented lines at header section.
        self.is_compressed: bool = 0                # Is main data compressed?

        # Date-time specifications
        self.start_date: datetime = None            # first reported date-time "YYYY-MM-DD HH:MM:SS"
        self.end_date: datetime = None              # last reported date-time  "YYYY-MM-DD HH:MM:SS"
        self.time_step: timedelta = None            # time step in seconds.
        self.run_length: int = 0                    # number of time steps on run.
        self.timezone: timedelta = timedelta(0)     # timezone

        # Species information
        self.nsp: int = 0                           # Number of species reported
        self.species: List[str] = []                # List of species with their properties

        # Proj specifications (in the future it should be just a projstring)
        self.proj: Dict = {
            'crs': 'UTM',                           # Coordinate system
            'zone': 0,                              # UTM zone 
            'hemis': '',                            # UTM zone 
            'datum': 'WGS84'                        # Datum
        }

        # Grid specifications ("sample" grid)
        self.nx = 0  ; self.ny = 0     # Grid size
        self.dx = 0.0; self.dy = 0.0   # Grid spacing 
        self.x0 = 0.0; self.y0 = 0.0   # Grid origin (south-west corner)

        # Receptors information
        self.receptor_type: int = 0                 # 0: gridded, 1: discrete, 2:complex-terrain
        self.gridded_receptors: bool = 0            # Is there a grid of receptors?

        self.ngrec: int = 0                         # Number of gridded receptors
        self.ndrec: int = 0                         # Number of discrete receptors
        self.nctrec: int = 0                        # Number of complex-terrain receptors

        self.rgroups: List[int]                     # list of all groups id
        self.igrp: Optional[np.ndarray]             # array with id-group (for discrete receptor)

        self.x: Optional[np.ndarray]
        self.y: Optional[np.ndarray]
        self.z: Optional[np.ndarray]                #ground elevation
        self.h: Optional[np.ndarray]                #above ground elevation
        self.ihill: Optional[np.ndarray]            #hill id group (for complex-terrain receptors)

        # Source information
        self.nsrctype: int = 0                      # Number of source types: POINT, LINE, ...
        self.nsrcbytype: List[int]                  # Number of sources of each type
        self.src_names: List[str] 

    def info(self):

        idate=self.start_date.strftime("%Y %j %H:%M:%S")
        edate=self.end_date.strftime("%Y %j %H:%M:%S")

        print(f"Calpuff Output File \"{self.filepath}\":")
        if (self.is_compressed): print(f"(Compressed)")
        print(f"{self.title}")
        print(f"   Start Date: {idate}.")
        print(f"   End   Date: {edate}.")
        print(f"   Time-Step:  {self.time_step}.")    
        print(f"   Run Lenght: {self.run_length}.") #Avg. Time: {self.time_ave}. 
        print(f"")
        print(f"   Grid parameters:")
        print(f"      Grid size:  {self.nx} x {self.ny}.")
        print(f"      Cell size:  {self.dx} x {self.dy} m.")
        print(f"      Xmin, Ymin: ({self.x0}, {self.y0})")
        print(f"")
        print(f"   Proj. parameters:\n       {self.proj['datum']}, {self.proj['crs']}, {self.proj['zone']} {self.proj['hemis']}")
        print(f"")
        print(f"   Species: ({self.nsp})")
        print(f"      Species names: {self.species}")
        print(f"      Species units: {self.units}")
        print(f"")
        print(f"   Receptors:")
        print(f"      Number of gridded    rec.: {self.ngrec} ")
        print(f"      Number of discrete   rec.: {self.ndrec} ")
        print(f"      Number of Cmpx Terr. rec.: {self.nctrec}")
        #print(f"      Number of receptor groups: {self.rgroups}")
        #print(f"      Receptor groups          : {self.igrp}")
        print(f"")
        print(f"   Sources:")
        print(f"      Number of source type   : {self.nsrctype}")
        print(f"      Number of source by type: {self.nsrcbytype}")
        print(f"      Source names:           : {self.src_names}")

    def get_coordinates(self):

        if ( self.gridded_receptors):

            nx = self.nx; ny = self.ny
            dx = self.dx; dy = self.dy
            x0 = self.x0; y0 = self.y0

            x = x0 + np.arange(nx) * dx
            y = y0 + np.arange(ny) * dy

            # create 2D meshgrid of coordinates
            X, Y = np.meshgrid(x, y) 

        elif ( self.ndrec > 0) :
            X = self.x; Y = self.y

        elif ( self.nctrec > 0 ):
            X = self.x; Y = self.y

        return X, Y

    def get_gridded_data(self, pollut):
         """
         Returns CALPUFF gridded receptors data.
         """
         if ( not self.gridded_receptors ):
             raise ValueError(f"No gridded receptors found in file {self.filepath}")
         if ( not pollut in self.species ):
             raise ValueError(f"No pollutant {pollut} found in file {self.filepath}")

         NT=self.run_length
         out = np.zeros([NT, self.ny, self.nx])

         with open(self.filepath, "rb") as f:

            #Skip header
            _skip_n_lines(f,7+self.NCOM)
            if ( self.ndrec > 0):
                _skip_n_lines(f, 2)          # Record #NCOM+6 : discrete receptors
                                             # Record #NCOM+6a: receptors group names
            if (self.nctrec > 0):
                _skip_n_lines(f,1)           # Record #NCOM+7 : complex terrain receptors    
            for i in range(self.nsrctype):
                n=self.nsrcbytype[i]
                if ( n > 0 ):
                    _skip_n_lines(f,1)       # Record #NCOM+8 : source names

            #Main data:
            for t in range(NT): #temporal loop
                #write(io8)nyrab,njulab,nhrab,nsecab,nyre,njule,nhre,nsece
                f.read(4)[0]
                yr1,dy1,hr1,sec1, yr2,dy2,hr2,sec2 = unpack("8i", f.read(32))
                f.read(4) #EOL
            
                #write(io8)ktype,ksource,csrcnam,xmapkm,ymapkm
                f.read(4)[0]
                ktype,ksource,csrcnam,xmapkm,ymapkm = unpack("2i 16s 2f", f.read(32))
                f.read(4)
            
                #Loop on output species
                for sp in range(self.nsp): 

                    # ---    Gridded receptor concentrations
                    if ( pollut == self.species[sp] ):
    
                        if ( self.is_compressed ):
                            f.read(4)
                            ii = unpack("i", f.read(4))[0] # "number of words" in compressed record.
                            f.read(4)
            
                            f.read(4)
                            specie = f.read(15).decode("utf-8").split() 
                            raw = list(unpack(f'{ii}f', f.read(4*ii) ))
                            xdat = _decompress( raw )
                            out[t,:,:] = np.array(xdat).reshape((self.ny, self.nx)) #, order='F')
                            f.read(4) #EOL

                        else:
                            f.read(4)
                            specie = f.read(15).decode("utf-8").split() 
                            out[t,:,:] = np.array([[ unpack("f",f.read(4)) for i in range(self.ny)] for j in range(self.nx)])
                            f.read(4) #EOL
 
                    else:
                        if (self.is_compressed):
                            _skip_n_lines(f, 2)
                        else:
                            _skip_n_lines(f, 1)

                    ## ---    Discrete receptor concentrations
                    if ( self.ndrec > 0):
                        if (self.is_compressed):
                            _skip_n_lines(f, 2)
                        else:
                            _skip_n_lines(f, 1)

                    ## ---    Discrete CTSG receptor concentrations
                    if ( self.nctrec > 0):
                        if (self.is_compressed):
                            _skip_n_lines(f, 2)
                        else:
                            _skip_n_lines(f, 1)

         return(out)

    def get_discrete_data(self, pollut):
         """
         Returns CALPUFF discrete receptors data.
         """
        
         if ( self.ndrec < 1):
             raise ValueError(f"No discrete receptors found in file {self.filepath}")
         if ( not pollut in self.species ):
             raise ValueError(f"No pollutant {pollut} found in file {self.filepath}")

         NT = self.run_length
         out = np.zeros([NT, self.ndrec ])
                                                                                             
         with open( self.filepath, "rb") as f:

            #Skip header
            _skip_n_lines(f,7+self.NCOM)
            if ( self.ndrec > 0):
                _skip_n_lines(f, 2)          # Record #NCOM+6 : discrete receptors
                                             # Record #NCOM+6a: receptors group names
            if (self.nctrec > 0):
                _skip_n_lines(f,1)           # Record #NCOM+7 : complex terrain receptors    

            for i in range(self.nsrctype):
                n=self.nsrcbytype[i]
                if ( n > 0 ):
                    _skip_n_lines(f,1)       # Record #NCOM+8 : source names

            #Main data:
            for t in range(NT): #temporal loop
                #write(io8)nyrab,njulab,nhrab,nsecab,nyre,njule,nhre,nsece
                f.read(4)[0]
                yr1,dy1,hr1,sec1, yr2,dy2,hr2,sec2 = unpack("8i", f.read(32))
                f.read(4) #EOL
                
                #write(io8)ktype,ksource,csrcnam,xmapkm,ymapkm
                f.read(4)[0]
                ktype,ksource,csrcnam,xmapkm,ymapkm = unpack("2i 16s 2f", f.read(32))
                f.read(4)

                #Loop on output species  
                for sp in range(self.nsp): 
                                                                                                                                 
                    # ---    Gridded receptor concentrations
                    if ( self.gridded_receptors ):
                        if ( pollut == self.species[sp] ):
                                                                                                                                 
                            if ( self.is_compressed ):
                                _skip_n_lines(f,2)

                            else:
                                _skip_n_lines(f,1)
                                                                                                                             
                    ## ---    Discrete receptor concentrations
                    if ( self.species[sp] == pollut ):

                        if ( self.is_compressed ):
                            f.read(4)
                            ii = unpack("i", f.read(4))[0] #
                            f.read(4)
                            f.read(4)
                            specie = f.read(15).decode("utf-8").split() 
                            raw = list(unpack(f'{ii}f', f.read(4*ii) ))
                            xdat = _decompress( raw )
                            out[t,:] = np.array(xdat) 
                            f.read(4) #EOL
                        else:
                            f.read(4)
                            specie   = unpack("15s",f.read(15))
                            for i in range(self.ndrec):
                                tmp= unpack("f",f.read(4))[0]
                                print(tmp)
                                out[t,i] = tmp
                            f.read(4)
                    else:
                        if (self.is_compressed):
                            _skip_n_lines(f, 2)
                        else:
                            _skip_n_lines(f, 1)

                    ### ---    Discrete CTSG receptor concentrations
                    if ( self.nctrec > 0):
                        if (self.is_compressed):
                            _skip_n_lines(f, 2)
                        else:
                            _skip_n_lines(f, 1)


         return(out)

    def get_data(self,pollut):

        if ( self.gridded_receptors):
            data = self.get_gridded_data(pollut) 
        elif ( self.ndrec > self.ngrec ):
            data = self.get_discrete_data(pollut)
        elif ( self.nctrec > self.ngrec ):
            raise ValueError(f"get_data not implemented yet for Complex terrain receptors.")
            #data = self.get_ct_data(pollut)
        else:
            raise ValueError(f"Not sufficient receptors found to make data extraction")

        return(data)

    def get_time_avg_max(self, pollut, interval, rank=1):
        """
        Computes, for each grid cell (of any rank), the maximum of the time-averaged fields.
    
        Parameters:
            pollut (str): Name of the pollutant.
            interval (int): Number of time steps to average over.
    
        Returns:
            np.ndarray: Array of shape [spatial_dims...] with the max of time-averaged values.
        """

        data = self.get_data(pollut)

        nt = data.shape[0]
        if nt % interval != 0:
            trimmed = nt - (nt % interval)
            data = data[:trimmed]
            print(f"Warning: data trimmed to {trimmed} time steps to match interval")
    
        # Calculate number of chunks
        n_chunks = data.shape[0] // interval
    
        # Reshape: [n_chunks, interval, ...]
        new_shape = (n_chunks, interval) + data.shape[1:]
        data_reshaped = data.reshape(new_shape)
    
        # Average over the interval axis (axis=1)
        averaged = data_reshaped.mean(axis=1)
        if rank > averaged.shape[0]:
            raise ValueError(f"Requested nth max ({n}) is larger than the number of averaged time blocks ({averaged.shape[0]})")
        
        # Use np.partition to find the nth maximum efficiently
        # The nth max is at index -n after sorting in descending order
        sorted_vals = np.sort(averaged, axis=0)[::-1]  # Descending sort
        nth_max = sorted_vals[rank - 1]  # n=1 means max
        
        ## Max over the time-averaged fields (axis=0)
        #max_field = averaged.max(axis=0)
        #return max_field

        return nth_max

def read_file(filepath: str) -> 'CalpuffOutput':
    """
    Opens and reads a CALPUFF output file, parsing all header information
    and initializing a CalpuffOutput object.

    Args:
        filepath: Path to the CALPUFF output file (e.g: CONC.DAT)

    Returns:
        Initialized CalpuffOutput object with all header information

    Raises:
        ValueError: If file format is invalid
        IOError: If file cannot be read
    """
    # Initialize empty CalpuffOutput object
    out = CalpuffOutput()
    out.filepath = filepath

    # Open the binary file
    with open(filepath, "rb") as f:
        print(f"Reading file {filepath}...")
        #Fortran's UNFORMATTED RECORDS have this structure :
        #  [4-byte length] [binary data] [4-byte length]

        # RECORD #1 --  
        #write(io8) conset,dataver,datamod
        _skip_n_lines(f,1)
    
        # RECORD #2 --  NCOM     
        #write(io8) ncom
        f.read(4) #record marker
        NCOM = unpack("i", f.read(4))[0]
        f.read(4) #end of record

        # RECORD #3 to NCOM+2 --  Comments.
        for _ in range(NCOM): _skip_n_lines(f,1) #f.seek(f.read(4)[0]+4, 1)#skip record 

        # RECORD #NCOM+3 -- General run parameters
        #write(io8)mmodel,ver,level,ibyr,ibjul,ibhr,ibsec,
        #1 abtz,irlg,iavg,nsecdt,nx,ny,dxkm,dykm,ione,xorigkm,yorigkm,
        #2 nssta(1),ibcomp,iecomp,jbcomp,jecomp,ibsamp,jbsamp,iesamp,
        #3 jesamp,meshdn,nsrctype,msource,ndrec,nrgrp,
        #4 nctrec,lsamp,nspout,lcomprs,i2dmet,
        #5 iutmzn,feast,fnorth,rnlat0,relon0,xlat1,xlat2,
        #6 pmap,utmhem,datum,daten,clat0,clon0,clat1,clat2
        f.read(4) 
        MODEL, VERSION, LEVEL          = unpack("12s 12s 12s", f.read(36))
        IBYR, IBJUL, IBHR, IBSEC, ABTZ = unpack("4i 8s",       f.read(24))
        IRLG, IAVG, NSECDT = unpack("3i", f.read(12))
        NX,NY, DXKM,DYKM, IONE, XORIGKM,YORIGKM  = unpack("2i 2f i 2f",f.read(28))

        nssta = unpack("i", f.read(4))                            #n ssurface stations
        IBCOMP, IECOMP, JBCOMP, JECOMP = unpack("4i", f.read(16)) #computational grid
        IBSAMP, JBSAMP, IESAMP, JESAMP = unpack("4i", f.read(16)) #sampling grid
        MESHDN = unpack("i", f.read(4))[0]                        #sampling grid spacing factor
    
        nsrctype, msource, ndrec, nrgrp, nctrec  = unpack("5i", f.read(20)) #
        lsamp, nspout, lcomprs, i2dmet          = unpack("4i", f.read(16))  #
        iutmzn, feast,fnorth,rnlat0,relon0,xlat1,xlat2 = unpack("i 6f", f.read(28))       # proj parameters.
        pmap,utmhem,datum,daten,clat0,clon0,clat1,clat2 = unpack("8s 4s 8s 12s 16s 16s 16s 16s", f.read(96)) # proj parameters.
        f.read(4) #close record

        #Definitions of Sample Grid (which is the one reported)
        out.nx = (IESAMP - IBSAMP ) * MESHDN +1
        out.ny = (JESAMP - JBSAMP ) * MESHDN +1
        out.dx = (DXKM / MESHDN) * 1000.
        out.dy = (DYKM / MESHDN) * 1000.
        out.x0 = (XORIGKM + IBSAMP * DXKM)* 1000
        out.y0 = (YORIGKM + JBSAMP * DYKM)* 1000

        # NON-REPORTED RECORD: how many sources from each type
        #write(io8) (nsrcbytype(n),n=1,nsrctype)
        f.read(4) #record marker
        nsrcbytype = unpack(str(nsrctype)+"i", f.read(nsrctype*4)) #[nsrcbytype(n),n=1,nsrctype]
        f.read(4) #close record
    
        # RECORD #NCOM+4 -- TITLE
        f.read(4) #record marker
        TITLE = f.read(80*3).decode("utf-8").strip()
        f.read(4) #EOL
    
        # RECORD #NCOM+5 -- List of species-groups output
        #write(io8)(csout(n),n=1,nspout)
        #for grp in range(ngrp):
        f.read(4)
        csout=[f.read(15).decode("utf-8")[0:11].strip() for _ in range(nspout)]  #1-12: specie's name, 13-15: layer
        f.read(4) #EOL
    
        # RECORD #NCOM+5a-- List of species-groups units 
        #write(io8)(acunit(n),n=1,nspout)
        f.read(4) # record marker
        acunit=[f.read(16).decode("utf-8").strip() for _ in range(nspout)]
        f.read(4) #EOL
    
        # RECORD #NCOM+6 -- Discrete (non-gridded) receptor data
        #write(io8)(tmp3(n),n=1,ndrec),
        #1     (tmp4(n2),n2=1,ndrec),(elevng(n3),n3=1,ndrec),
        #2     (zng(n4),n4=1,ndrec), (irgrp(n5),n5=1,ndrec)
        if (ndrec > 0):
            f.read(4) #record marker
            out.x = np.array([ unpack("f",f.read(4)) for _ in range(ndrec)])*1e3 #km to m
            out.y = np.array([ unpack("f",f.read(4)) for _ in range(ndrec)])*1e3 #km to m
            out.z = np.array([ unpack("f",f.read(4)) for _ in range(ndrec)])
            out.h = np.array([ unpack("f",f.read(4)) for _ in range(ndrec)])
            out.igrp= np.array([ unpack("i",f.read(4)) for _ in range(ndrec)])
            f.read(4) #EOL
    
            # RECORD #NCOM+6a -- Receptor-group names
            rgrpnam=[""]*nrgrp
            f.read(4) #record marker
            for i in range(nrgrp):
                rgrpnam[i] = f.read(80).decode("utf-8").strip()
            f.read(4) #EOL
            out.rgroups = rgrpnam

    
        # RECORD #NCOM+7 -- Complex terrain receptor data
        if (nctrec > 0):
            f.read(4) #record marker
            out.x = np.array([ unpack("f",f.read(4)) for _ in range(nctrec)])*1e3 #km to m
            out.y = np.array([ unpack("f",f.read(4)) for _ in range(nctrec)])*1e3 #km to m
            out.z = np.array([ unpack("f",f.read(4)) for _ in range(nctrec)])
            out.ihill=np.array([ unpack("i",f.read(4)) for _ in range(nctrec)])
            f.read(4) #EOL
            
        # RECORD #NCOM+8 -- Source names
        #do itype=1,nsrctype
        #if(nsrcbytype(itype).GT.0) then
        #write(io8) itype,[cnamsrc(n,itype), n=1,nsrcbytype(itype)]
        #print(f"Number of sorces types: {nsrctype}")
        cnamsrc=[""]*nsrctype
        for itype in range(nsrctype):
            n=nsrcbytype[itype] 
            if ( n > 0 ):
                f.read(4) #open record line
                f.read(4) #itype      
                cnamsrc = [f.read(16).decode("utf-8").strip() for j in range(n)]
                f.read(4) #EOL
    
        # RECORD #NCOM+8  -- Nearest Surface Station   for VISIBILITY ONLY
        # RECORD #NCOM+9  -- X coord (UTM) of stations for VISIBILITY ONLY
        # RECORD #NCOM+10 -- Y coord (UTM) of stations for VISIBILITY ONLY

        # Temporal params:
        out.model_version=VERSION
        out.start_date=datetime.strptime(f"{IBYR} {IBJUL} {IBHR} {IBSEC}","%Y %j %H %S")
        out.run_length=IRLG
        out.ave_time =timedelta(hours=IAVG)
        out.time_step=timedelta(seconds=NSECDT)
        out.end_date=out.start_date+out.time_step*out.run_length

        # Proj params:
        out.proj['crs']   = pmap.decode("utf-8").strip()     #
        out.proj['datum'] = datum.decode("utf-8").strip()    #
        out.proj['zone']  = iutmzn                            
        out.proj['hemis'] = utmhem.decode("utf-8").strip()   # north/south-ern hemisphere                                                  

        # Species params:
        out.nsp     = nspout
        out.species = csout
        out.units   = acunit
                                                                                            
        # Receptors params:
        out.gridded_receptors = lsamp    # gridded receptors
        out.ngrec   = out.nx*out.ny      # num. of gridded         receptors
        out.ndrec   = ndrec              # num. of discrete        receptors
        out.nctrec  = nctrec             # num. of complex-terrain receptors

        out.nrgrp   = nrgrp              # Number of receptor groups

        # Sources params:
        out.nsrctype=nsrctype
        out.nsrcbytype=nsrcbytype
        out.src_names=cnamsrc

        # Internal:
        out.is_compressed = lcomprs
        out.NCOM = NCOM

        return(out)

