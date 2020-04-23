# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:08:36 2020

@author: jpeacock
"""

from dateutil import parser as dt_parser
from mtpy.utils import gis_tools

# ==============================================================================
# Station details
# ==============================================================================
class Station(object):
    """
    Information on the station, including location, id, etc.

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    aqcuired_by       string       name of company or person whom aqcuired the
                                   data.
    id                string       station name
    Location          object       Holds location information, lat, lon, elev
                      Location     datum, easting, northing see Location class
    start_date        string       YYYY-MM-DD start date of measurement
    end_date          string       YYYY-MM-DD end date of measurement
    year_collected    string       year data collected
    survey            string       survey name
    project           string       project name
    run_list          string       list of measurment runs ex. [mt01a, mt01b]
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> Site(**{'state':'Nevada', 'Operator':'MTExperts'})

    """

    def __init__(self, **kwargs):
        self.acquired_by = None
        self._end_dt = None
        self.id = None
        self.Location = Location()
        self.project = None
        self.run_list = None
        self._start_dt = None
        self.survey = None
        self._date_fmt = '%Y-%m-%d'

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])
            
    @property
    def year_collected(self):
        try:
            return self._start_dt.year
        except TypeError:
            return None
    @year_collected.setter
    def year_collected(self, value):
        pass
    
    @property
    def start_date(self):
        try:
            return self._start_dt.strftime(self._date_fmt)
        except AttributeError:
            return None
        
    @start_date.setter
    def start_date(self, start_str):
        self._start_dt = self._read_date(start_str)
            
    @property
    def end_date(self):
        try:
            return self._end_dt.strftime(self._date_fmt)
        except AttributeError:
            return None
        
    @end_date.setter
    def end_date(self, end_str):
        self._end_dt = self._read_date(end_str)
            
    def _read_date(self, date_str):
        """
        read a date string
        """
        if date_str in [None, 'None', 'none', 'NONE']:
            return None
        try:
            return dt_parser.parse(date_str)
        except dt_parser.ParserError:
            try:
                return dt_parser.parse(date_str, dayfirst=True)
            except dt_parser.ParserError as error:
                raise ValueError(error)
        
# ==============================================================================
# Location class, be sure to put locations in decimal degrees, and note datum
# ==============================================================================
class Location(object):
    """
    location details
    """

    def __init__(self, **kwargs):
        self.datum = 'WGS84'
        self.declination = None
        self.declination_epoch = None

        self._elevation = None
        self._latitude = None
        self._longitude = None

        self._northing = None
        self._easting = None
        self.utm_zone = None
        self.elev_units = 'm'
        self.coordinate_system = 'Geographic North'

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, lat):
        self._latitude = gis_tools.assert_lat_value(lat)

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, lon):
        self._longitude = gis_tools.assert_lon_value(lon)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, elev):
        self._elevation = gis_tools.assert_elevation_value(elev)

    @property
    def easting(self):
        return self._easting

    @easting.setter
    def easting(self, easting):
        try:
            self._easting = float(easting)
        except TypeError:
            self._easting = None

    @property
    def northing(self):
        return self._northing

    @northing.setter
    def northing(self, northing):
        try:
            self._northing = float(northing)
        except TypeError:
            self._northing = None

    def project_location2utm(self):
        """
        project location coordinates into meters given the reference ellipsoid,
        for now that is constrained to WGS84

        Returns East, North, Zone
        """
        utm_point = gis_tools.project_point_ll2utm(self._latitude,
                                                   self._longitude,
                                                   datum=self.datum)

        self.easting = utm_point[0]
        self.northing = utm_point[1]
        self.utm_zone = utm_point[2]

    def project_location2ll(self):
        """
        project location coordinates into meters given the reference ellipsoid,
        for now that is constrained to WGS84

        Returns East, North, Zone
        """
        ll_point = gis_tools.project_point_utm2ll(self.easting,
                                                  self.northing,
                                                  self.utm_zone,
                                                  datum=self.datum)

        self.latitude = ll_point[0]
        self.longitude = ll_point[1]

# =============================================================================
# Channel class
# =============================================================================
class Channel(object):
    """
    Class to hold information about a channel whether it be electric or 
    magnetic.
    
    
    """
    
    def __init__(self, **kwargs):
        
        self.type = None
        self.azimuth = None
        self.dip = None
        self.location = Location()
        self.dipole_length = None
        self.channel_number = None
        self.sensor = Instrument()
        self.units = None
        
        
        


# ==============================================================================
# Field Notes
# ==============================================================================
class FieldNotes(object):
    """
    Field note information.


    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    data_quality      DataQuality notes on data quality
    electrode         Instrument      type of electrode used
    data_logger       Instrument      type of data logger
    magnetometer      Instrument      type of magnetotmeter
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> FieldNotes(**{'electrode_ex':'Ag-AgCl 213', 'magnetometer_hx':'102'})
    """

    def __init__(self, **kwargs):

        self.DataQuality = DataQuality()
        self.DataLogger = Instrument()

        self.Electrode_ex = Channel(**{'type':'ex'})
        self.Electrode_ey = Channel(**{'type':'ey'})
        self.Magnetometer = Channel(**{'type':'hx'})
        self.Magnetometer_hy = Channel(**{'type':'hy'})
        self.Magnetometer_hz = Channel(**{'type':'hz'})

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Instrument
# ==============================================================================
class Instrument(object):
    """
    Information on an instrument that was used.

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    id                string      serial number or id number of data logger
    manufacturer      string      company whom makes the instrument
    type              string      Broadband, long period, something else
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> Instrument(**{'ports':'5', 'gps':'time_stamped'})
    """

    def __init__(self, **kwargs):
        self.id = None
        self.manufacturer = None
        self.type = None

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Data Quality
# ==============================================================================
class DataQuality(object):
    """
    Information on data quality.

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    comments          string      comments on data quality
    good_from_period  float       minimum period data are good
    good_to_period    float       maximum period data are good
    rating            int         [1-5]; 1 = poor, 5 = excellent
    warrning_comments string      any comments on warnings in the data
    warnings_flag     int         [0-#of warnings]
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>>DataQuality(**{'time_series_comments':'Periodic Noise'})
    """

    def __init__(self, **kwargs):
        self.comments = None
        self.good_from_period = None
        self.good_to_period = None
        self.rating = None
        self.warnings_comments = None
        self.warnings_flag = 0
        self.author = None

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Citation
# ==============================================================================
class Citation(object):
    """
    Information for a citation.

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    author            string      Author names
    title             string      Title of article, or publication
    journal           string      Name of journal
    doi               string      DOI number (doi:10.110/sf454)
    year              int         year published
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> Citation(**{'volume':56, 'pages':'234--214'})
    """

    def __init__(self, **kwargs):
        self.author = None
        self.title = None
        self.journal = None
        self.volume = None
        self.doi = None
        self.year = None

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Copyright
# ==============================================================================
class Copyright(object):
    """
    Information of copyright, mainly about how someone else can use these
    data. Be sure to read over the conditions_of_use.

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    citation          Citation    citation of published work using these data
    conditions_of_use string      conditions of use of these data
    release_status    string      release status [ open | public | proprietary]
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> Copyright(**{'owner':'University of MT', 'contact':'Cagniard'})
    """

    def __init__(self, **kwargs):
        self.Citation = Citation()
        self.conditions_of_use = ''.join(['All data and metadata for this survey are ',
                                          'available free of charge and may be copied ',
                                          'freely, duplicated and further distributed ',
                                          'provided this data set is cited as the ',
                                          'reference. While the author(s) strive to ',
                                          'provide data and metadata of best possible ',
                                          'quality, neither the author(s) of this data ',
                                          'set, not IRIS make any claims, promises, or ',
                                          'guarantees about the accuracy, completeness, ',
                                          'or adequacy of this information, and expressly ',
                                          'disclaim liability for errors and omissions in ',
                                          'the contents of this file. Guidelines about ',
                                          'the quality or limitations of the data and ',
                                          'metadata, as obtained from the author(s), are ',
                                          'included for informational purposes only.'])
        self.release_status = None
        self.additional_info = None
        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Provenance
# ==============================================================================
class Provenance(object):
    """
    Information of the file history, how it was made

    Holds the following information:

    ====================== =========== ========================================
    Attributes             Type        Explanation
    ====================== =========== ========================================
    creation_time          string      creation time of file YYYY-MM-DD,hh:mm:ss
    creating_application   string      name of program creating the file
    creator                Person      person whom created the file
    submitter              Person      person whom is submitting file for
                                       archiving
    ====================== =========== ========================================

    More attributes can be added by inputing a key word dictionary

    >>> Provenance(**{'archive':'IRIS', 'reprocessed_by':'grad_student'})
    """

    def __init__(self, **kwargs):
        self.creation_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        self.creating_application = 'MTpy'
        self.Creator = Person()
        self.Submitter = Person()

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Person
# ==============================================================================
class Person(object):
    """
    Information for a person

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    email             string      email of person
    name              string      name of person
    organization      string      name of person's organization
    organization_url  string      organizations web address
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> Person(**{'phone':'650-888-6666'})
    """

    def __init__(self, **kwargs):
        self.email = None
        self.name = None
        self.organization = None
        self.organization_url = None

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


# ==============================================================================
# Processing
# ==============================================================================
class Processing(object):
    """
    Information for a processing

    Holds the following information:

    ================= =========== =============================================
    Attributes         Type        Explanation
    ================= =========== =============================================
    email             string      email of person
    name              string      name of person
    organization      string      name of person's organization
    organization_url  string      organizations web address
    ================= =========== =============================================

    More attributes can be added by inputing a key word dictionary

    >>> Person(**{'phone':'888-867-5309'})
    """

    def __init__(self, **kwargs):
        self.Software = Software()
        self.notes = None
        self.processed_by = None
        self.sign_convention = 'exp(+i \omega t)'
        self.remote_reference = None
        self.RemoteSite = Site()

        for key in list(kwargs.keys()):
            setattr(self, key, kwargs[key])


class Software(object):
    """
    software
    """

    def __init__(self, **kwargs):
        self.name = None
        self.version = None
        self.Author = Person()

        for key in kwargs:
            setattr(self, key, kwargs[key])
