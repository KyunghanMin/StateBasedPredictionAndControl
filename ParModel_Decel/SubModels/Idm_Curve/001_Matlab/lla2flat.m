function ret = lla2flat( lla, llo, psio, href )
    % WSG84
    
    R = 6378137.0;  % Equator radius in meters
    f = 0.00335281066474748071;  % 1/298.257223563, inverse flattening
    
    Lat_p = lla(1) * pi / 180.0;  % from degrees to radians
    Lon_p = lla(2) * pi / 180.0;  % from degrees to radians
    Alt_p = lla(3);  % meters

    % Reference location (lat, lon), from degrees to radians
    Lat_o = llo(1) * pi / 180.0;
    Lon_o = llo(2) * pi / 180.0;
    
    psio = psio * pi / 180.0;

    dLat = Lat_p - Lat_o;
    dLon = Lon_p - Lon_o;

    ff = (2.0 * f) - (f ^ 2);  % Can be precomputed

    sinLat = sin(Lat_o);
    
    % Radius of curvature in the prime vertical
    Rn = R / sqrt(1 - (ff * (sinLat ^ 2)));
    
    % Radius of curvature in the meridian
    Rm = Rn * ((1 - ff) / (1 - (ff * (sinLat ^ 2))));
        
    dNorth = (dLat) / atan2(1, Rm);
    dEast = (dLon) / atan2(1, (Rn * cos(Lat_o)));
            
    % Rotate matrice clockwise
    Xp = (dNorth * cos(psio)) + (dEast * sin(psio));
    Yp = (-dNorth * sin(psio)) + (dEast * cos(psio));
    Zp = -Alt_p - href;
    
    Yp = -Yp;
    
    ret = [Xp, Yp, Zp];
end
