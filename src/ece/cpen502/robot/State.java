package ece.cpen502.robot;

public class State {

    public enum EnergyLabel {
        LOW, MEDIUM, HIGH
    }
    public enum DistanceLabel {
        VERY_CLOSE, NEAR, FAR
    }

    public int indexForEnergy(double energy) {
        EnergyLabel label = energyTransfer(energy);
        if (label == EnergyLabel.LOW) return 0;
        else if (label == EnergyLabel.MEDIUM) return 1;
        else if (label == EnergyLabel.HIGH) return 2;
        else return -1;
    }

    public int indexForDistance(double distance) {
        DistanceLabel label = distanceTransfer(distance);
        if (label == DistanceLabel.VERY_CLOSE) return 0;
        else if (label == DistanceLabel.NEAR) return 1;
        else if (label == DistanceLabel.FAR) return 2;
        else return -1;
    }

    public EnergyLabel energyTransfer(double energy) {
        if (energy <= 33.3) return EnergyLabel.LOW;
        else if (energy <= 66.7) return EnergyLabel.MEDIUM;
        else return EnergyLabel.HIGH;
    }

    public DistanceLabel distanceTransfer(double distance) {
        if (distance <= 100) return DistanceLabel.VERY_CLOSE;
        else if (distance <= 300) return DistanceLabel.NEAR;
        else return DistanceLabel.FAR;
    }
}
