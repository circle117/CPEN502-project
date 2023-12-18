package ece.cpen502.robot;

import ece.cpen502.robot.microBot.MicroBot;
import robocode.ScannedRobotEvent;

public class Action {

    public enum ActionLabel {
        CIRCLE, ADVANCE, HEAD2CENTER,
        FIRE, RETREAT
    }

    private ActionLabel label;


    /**
     * perform actions
     */
    public void doAction(MicroBot robot, ScannedRobotEvent event) {
        switch (label) {
            case CIRCLE:
                if (robot.getVelocity() == 0)
                    robot.setMoveDirection((byte) (robot.getMoveDirection() * -1));
                robot.setTurnRight(robot.normalizeBearing(robot.getEnemy().getBearing()+90));
                robot.setAhead(1000 * robot.getMoveDirection());
                break;
            case ADVANCE:
                robot.setAhead(event.getDistance()-10);
                break;
            case HEAD2CENTER:
                double centerAngle = Math.atan2(robot.getBattleFieldWidth()/2- robot.getX(),
                        robot.getBattleFieldHeight()/2-robot.getY());
                centerAngle = centerAngle * 360/ (2*Math.PI);
                robot.setTurnRight(robot.normalizeBearing(centerAngle - robot.getHeading()));
                double distance = Math.sqrt(Math.pow(robot.getBattleFieldWidth()/2 - robot.getX(),2) +
                                            Math.pow(robot.getBattleFieldHeight()/2 - robot.getY(), 2));
                robot.setAhead(distance);
                break;
            case FIRE:
                // it's a good idea to fire low-strength bullets when your enemy is far away,
                // and high-strength bullets when he's close.
                if (robot.getGunHeat() == 0 && Math.abs(robot.getGunTurnRemaining()) < 10)
                    robot.setFire(Math.min(robot.getFireVariable()/ robot.getEnemy().getDistance(), 3));
                break;
            case RETREAT:
                robot.setBack(300);
                break;
        }
    }

    /**
     * map index to action Enum
     */
    public ActionLabel labelTransfer(double index) {
        if (index == 0.0) return ActionLabel.CIRCLE;
        else if (index == 1.0) return ActionLabel.ADVANCE;
        else if (index == 2.0) return ActionLabel.HEAD2CENTER;
        else if (index == 3.0) return ActionLabel.FIRE;
        else if (index == 4.0) return ActionLabel.RETREAT;
        else return null;
    }

    public void setLabel(ActionLabel label) {
        this.label = label;
    }
}
